# Copyright 2020 The Lucent Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import, division, print_function

import warnings
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch

from lucent.optvis import objectives, transform, param
from lucent.misc.io import show


def pixelwise_gradient_change_interrupt(_optimizer, _params, smooth_pixelwise_grads, winsize=2000, target_ratio=1.1):
    """
    Interrupts the optimization if the relative change in pixelwise grads in a certain window was below threshold.
    """

    # just continue if we haven't even collected enough values
    if len(smooth_pixelwise_grads) < winsize:
        return False

    first_half = np.mean(smooth_pixelwise_grads[-winsize:-winsize//2])
    second_half = np.mean(smooth_pixelwise_grads[-winsize//2:])

    ratio = first_half / second_half

    return ratio <= target_ratio


def render_vis(
    model,
    objective_f,
    param_f=None,
    optimizer=None,
    transforms=None,
    thresholds=(512,),
    verbose=False,
    preprocess=True,
    progress=True,
    show_image=True,
    save_image=False,
    image_name=None,
    show_inline=False,
    fixed_image_size=None,
    # additional parameters
    layer=None,
    channel=0,
    min_steps=2500,
    interrupt_interval=0,
    interrupt_condition=None,
    smoothing_weight=0.99,
    return_smooth_grads=False
):
    """
    Adapted version of render_vis that supports interrupting the optimization procedure when a condition is fulfilled.
    layer and channel are the layer and channel of the targeted unit TODO get this from the objective?
    min_steps is the minimum number of steps that should always be made, irrespective of gradient norm
    interrupt_interval is the number of steps between evaluating the condition
    interrupt_condition is a function that takes optimizer, params, activations and mean absolute gradients in image-space and returns true if the optimization should be interrupted
    smoothing_weight is the weight used in the exponential smoothing of mean gradients
    if return_smooth_grads is True, not only the images but also the sequence of gradients is returned
    """

    if param_f is None:
        param_f = lambda: param.image(128)
    # param_f is a function that should return two things
    # params - parameters to update, which we pass to the optimizer
    # image_f - a function that returns an image as a tensor
    params, image_f = param_f()
    assert len(params[0].shape) == 5, "Only works for optimization in Fourier space!" # no other way of telling, param_f could be anything

    if optimizer is None:
        optimizer = lambda params: torch.optim.Adam(params, lr=5e-2)
    optimizer = optimizer(params)

    if transforms is None:
        transforms = transform.standard_transforms
    transforms = transforms.copy()

    if preprocess:
        if model._get_name() == "InceptionV1":
            # Original Tensorflow InceptionV1 takes input range [-117, 138]
            transforms.append(transform.preprocess_inceptionv1())
        else:
            # Assume we use normalization for torchvision.models
            # See https://pytorch.org/docs/stable/torchvision/models.html
            transforms.append(transform.normalize())

    # Upsample images smaller than 224
    image_shape = image_f().shape
    if fixed_image_size is not None:
        new_size = fixed_image_size
    elif image_shape[2] < 224 or image_shape[3] < 224:
        new_size = 224
    else:
        new_size = None
    if new_size:
        transforms.append(
            torch.nn.Upsample(size=new_size, mode="bilinear", align_corners=True)
        )

    transform_f = transform.compose(transforms)

    @torch.no_grad()
    def get_mean_px_grad(fft_grads):

        if type(fft_grads) is not torch.complex64:
            fft_grads = torch.view_as_complex(fft_grads)

        _, _, h, w = image_shape
        # NOTE in lucent.param.spatial.fft_image(), they scale the spectrum first
        ifft = torch.fft.irfftn(fft_grads, s=(h,w), norm='ortho')

        # always look at the max of the batch, to make sure that even the worst image is okay
        return torch.norm(ifft, p=2, dim=(1,2,3)).max().detach().cpu().numpy()

    with ModelHook(model, image_f) as hook:
        objective_f = objectives.as_objective(objective_f)

        if verbose:
            model(transform_f(image_f()))
            print("Initial loss: {:.3f}".format(objective_f(hook)))

        images = []
        smooth_pixelwise_grads = []
        try:
            for i in tqdm(range(1, max(thresholds) + 1), disable=(not progress)):
                def closure():
                    optimizer.zero_grad()
                    try:
                        model(transform_f(image_f()))
                    except RuntimeError as ex:
                        if i == 1:
                            # Only display the warning message
                            # on the first iteration, no need to do that
                            # every iteration
                            warnings.warn(
                                "Some layers could not be computed because the size of the "
                                "image is not big enough. It is fine, as long as the non"
                                "computed layers are not used in the objective function"
                                f"(exception details: '{ex}')"
                            )
                    loss = objective_f(hook)
                    loss.backward()
                    return loss

                optimizer.step(closure)

                acts = hook(layer)

                # get mean for each channel for conv-layers
                if len(acts.shape) > 2:
                    acts = torch.mean(acts, dim=(-1, -2)) # NxC

                # calculate mean absolute pixel-wise change in image-space
                mean_px_grad = get_mean_px_grad(params[0].grad)

                # calculate exponentially smoothed value and append to list
                last = smooth_pixelwise_grads[-1] if smooth_pixelwise_grads else mean_px_grad
                smoothed_val = last * smoothing_weight + (1 - smoothing_weight) * mean_px_grad
                smooth_pixelwise_grads.append(smoothed_val)

                if i in thresholds:
                    image = tensor_to_img_array(image_f())
                    if verbose:
                        print("Loss at step {}: {:.3f}".format(i, objective_f(hook)))
                        if show_inline:
                            show(image)
                    images.append(image)

                if interrupt_condition is not None and i > min_steps and i % interrupt_interval == 0:
                    if interrupt_condition(optimizer, params, smooth_pixelwise_grads):
                        if verbose:
                            print(f"Interrupting due to condition at step {i}")
                        image = tensor_to_img_array(image_f())
                        images.append(image)
                        break

        except KeyboardInterrupt:
            print("Interrupted optimization at step {:d}.".format(i))
            if verbose:
                print("Loss at step {}: {:.3f}".format(i, objective_f(hook)))
            images.append(tensor_to_img_array(image_f()))

    if save_image:
        export(image_f(), image_name)
    if show_inline:
        show(tensor_to_img_array(image_f()))
    elif show_image:
        view(image_f())

    if return_smooth_grads:
        return images, smooth_pixelwise_grads
    
    return images

def tensor_to_img_array(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, [0, 2, 3, 1])
    return image


def view(tensor):
    image = tensor_to_img_array(tensor)
    assert len(image.shape) in [
        3,
        4,
    ], "Image should have 3 or 4 dimensions, invalid image shape {}".format(image.shape)
    # Change dtype for PIL.Image
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 4:
        image = np.concatenate(image, axis=1)
    Image.fromarray(image).show()


def export(tensor, image_name=None):
    image_name = image_name or "image.jpg"
    image = tensor_to_img_array(tensor)
    assert len(image.shape) in [
        3,
        4,
    ], "Image should have 3 or 4 dimensions, invalid image shape {}".format(image.shape)
    # Change dtype for PIL.Image
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 4:
        image = np.concatenate(image, axis=1)
    Image.fromarray(image).save(image_name)


class ModuleHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = None
        self._features = dict()

    @property
    def features(self):
        keys = list(sorted(self._features.keys()))
        if len(keys) == 1:
            return self._features[keys[0]]
        else:
            return torch.nn.parallel.gather([self._features[k] for k in keys], keys[0])

    def hook_fn(self, module, input, output):
        self.module = module
        device = output.device
        self._features[str(device)] = output

    def close(self):
        self.hook.remove()


class ModelHook:
    def __init__(self, model, image_f=None, layer_names=None):
        self.model = model
        self.image_f = image_f
        self.features = {}
        self.layer_names = layer_names
       
    def __enter__(self):
        # recursive hooking function
        def hook_layers(net, prefix=[]):
            if hasattr(net, "_modules"):
                layers = list(net._modules.items())
                for i, (name, layer) in enumerate(layers):
                    if layer is None:
                        # e.g. GoogLeNet's aux1 and aux2 layers
                        continue

                    if self.layer_names is not None and i < len(layers) - 1:
                        # only save activations for chosen layers
                        if name not in self.layer_names:
                            continue
                    
                    self.features["_".join(prefix + [name])] = ModuleHook(layer)
                    hook_layers(layer, prefix=prefix + [name])

        if isinstance(self.model, torch.nn.DataParallel):   
            hook_layers(self.model.module)
        else:
            hook_layers(self.model)
        
        def hook(layer):
            if layer == "input":
                out = self.image_f()
            elif layer == "labels":
                out = list(self.features.values())[-1].features
            else:
                assert layer in self.features, f"Invalid layer {layer}. Retrieve the list of layers with `lucent.modelzoo.util.get_model_layers(model)`."
                out = self.features[layer].features
            assert out is not None, "There are no saved feature maps. Make sure to put the model in eval mode, like so: `model.to(device).eval()`. See README for example."
            return out

        return hook
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for k in self.features.copy():
            self.features[k].close()
            del self.features[k]
        
        
