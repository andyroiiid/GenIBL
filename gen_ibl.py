# Copyright 2026 Andrew Huang. All Rights Reserved.

from pathlib import Path
import struct

from matplotlib import pyplot as plt
import numpy as np
import OpenImageIO as oiio
from tqdm import tqdm

from HaruPy import Glm, Spv, Vk
from HaruPy.Vk.Enums import *


def ceildiv(a: int, b: int) -> int:
    return -(a // -b)


def create_compute_pipeline(module_name, entry_point_name) -> Vk.ComputePipeline:
    spv = Spv.compile_shader(module_name, entry_point_name)
    shader = Vk.ShaderModule(spv)
    return Vk.ComputePipeline(shader)


def show_image(pixels, title):
    plt.title(title)
    plt.imshow(np.clip(pixels[:, :, :3], 0, 1))
    plt.tight_layout()
    plt.show()


def load_image(filename: str) -> np.ndarray:
    image = oiio.ImageInput.open(filename)
    spec = image.spec()
    pixels = image.read_image(spec.format)
    image.close()
    return pixels


def save_image(filename: str, pixels: np.ndarray):
    image = oiio.ImageOutput.create(filename)
    assert pixels.dtype == np.float32
    assert pixels.ndim == 3
    height, width, channels = pixels.shape
    spec = oiio.ImageSpec(width, height, channels, oiio.HALF)
    image.open(filename, spec)
    image.write_image(pixels)
    image.close()


class ImmediateContext:
    def __init__(self, timeout: int = 1_000_000_000):
        self.cmd_pool = Vk.CommandPool(VK_COMMAND_POOL_CREATE_TRANSIENT_BIT)
        self.cmd = Vk.CommandBuffer(self.cmd_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY)
        self.fence = Vk.Fence(0)
        self.timeout = timeout

    def __enter__(self):
        self.cmd.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
        return self.cmd

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cmd.end()
        self.cmd.submit(self.fence)
        self.fence.wait(self.timeout)


class BytesBuilder:
    def __init__(self):
        self.data = bytearray()

    def add(self, fmt: str, /, *v):
        self.data.extend(struct.pack(fmt, *v))
        return self

    def build(self):
        return bytes(self.data)


def create_image(pixels: np.ndarray) -> Vk.Image2D:
    assert pixels.dtype == np.float32
    assert pixels.ndim == 3
    height, width, channels = pixels.shape
    assert channels == 3 or channels == 4

    if channels == 3:
        padded = np.zeros((height, width, 4), dtype=np.float32)
        padded[:, :, :3] = pixels
        pixels = padded

    upload_buf = Vk.Buffer(pixels.size * pixels.itemsize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, Vk.HostAccess.Write)
    upload_buf.upload(pixels.tobytes())

    image = Vk.Image2D(
        format=VK_FORMAT_R32G32B32A32_SFLOAT,
        size=Glm.UVec2(width, height),
        mip_levels=1,
        usage=VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
    )

    with ImmediateContext() as cmd:
        cmd.buffer_barrier(upload_buf, Vk.BufferState.HostWrite, Vk.BufferState.TransferRead)
        cmd.image_barrier(image, Vk.ImageState.Undefined, Vk.ImageState.TransferDst)
        cmd.copy_buffer_to_image(upload_buf, 0, image, 0)
        cmd.image_barrier(image, Vk.ImageState.TransferDst, Vk.ImageState.ShaderReadOnly)

    return image


def prefilter_irradiance(image: Vk.Image2D, size: Glm.UVec2) -> Vk.Image2D:
    sampler = Vk.Sampler(
        mag_filter=VK_FILTER_LINEAR,
        min_filter=VK_FILTER_LINEAR,
        mipmap_mode=VK_SAMPLER_MIPMAP_MODE_LINEAR,
        address_mode_u=VK_SAMPLER_ADDRESS_MODE_REPEAT,
        address_mode_v=VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
        address_mode_w=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    )

    irradiance = Vk.Image2D(
        format=image.get_format(),
        size=size,
        mip_levels=1,
        usage=VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
    )

    pipeline = create_compute_pipeline("PrefilterIrradiance", "MainCS")

    with ImmediateContext() as cmd:
        cmd.image_barrier(irradiance, Vk.ImageState.Undefined, Vk.ImageState.General)

    for y in tqdm(range(size.y), "prefilter_irradiance"):
        with ImmediateContext() as cmd:
            cmd.enable_bindless()
            cmd.bind_compute_pipeline(pipeline)
            cmd.push_constants(
                BytesBuilder()
                .add("I", sampler.ensure_descriptor_id())
                .add("I", image.ensure_sampled_id())
                .add("I", irradiance.ensure_storage_id())
                .add("II", size.x, size.y)
                .add("I", y)
                .build()
            )
            cmd.dispatch(Glm.UVec3(ceildiv(size.x, 32), 1, 1))

    return irradiance


def prefilter_radiance(image: Vk.Image2D, size: Glm.UVec2, roughness: float) -> Vk.Image2D:
    sampler = Vk.Sampler(
        mag_filter=VK_FILTER_LINEAR,
        min_filter=VK_FILTER_LINEAR,
        mipmap_mode=VK_SAMPLER_MIPMAP_MODE_LINEAR,
        address_mode_u=VK_SAMPLER_ADDRESS_MODE_REPEAT,
        address_mode_v=VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
        address_mode_w=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    )

    radiance = Vk.Image2D(
        format=image.get_format(),
        size=size,
        mip_levels=1,
        usage=VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
    )

    pipeline = create_compute_pipeline("PrefilterRadiance", "MainCS")

    with ImmediateContext() as cmd:
        cmd.image_barrier(radiance, Vk.ImageState.Undefined, Vk.ImageState.General)

    for y in tqdm(range(size.y), f"prefilter_radiance(roughness={roughness:.2f})"):
        with ImmediateContext() as cmd:
            cmd.enable_bindless()
            cmd.bind_compute_pipeline(pipeline)
            cmd.push_constants(
                BytesBuilder()
                .add("I", sampler.ensure_descriptor_id())
                .add("I", image.ensure_sampled_id())
                .add("I", radiance.ensure_storage_id())
                .add("II", size.x, size.y)
                .add("I", y)
                .add("f", roughness)
                .build()
            )
            cmd.dispatch(Glm.UVec3(ceildiv(size.x, 32), 1, 1))

    return radiance


def integrate_brdf(size: int) -> Vk.Image2D:
    output = Vk.Image2D(
        format=VK_FORMAT_R32G32B32A32_SFLOAT,
        size=Glm.UVec2(size, size),
        mip_levels=1,
        usage=VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
    )

    pipeline = create_compute_pipeline("IntegrateBRDF", "MainCS")

    with ImmediateContext() as cmd:
        cmd.image_barrier(output, Vk.ImageState.Undefined, Vk.ImageState.General)

    for y in tqdm(range(size), "integrate_brdf"):
        with ImmediateContext() as cmd:
            cmd.enable_bindless()
            cmd.bind_compute_pipeline(pipeline)
            cmd.push_constants(
                BytesBuilder()
                .add("I", output.ensure_storage_id())
                .add("II", size, size)
                .add("I", y)
                .build()
            )
            cmd.dispatch(Glm.UVec3(ceildiv(size, 32), 1, 1))

    return output


def dump_image(image: Vk.Image2D) -> np.ndarray:
    image_size = image.get_size()
    assert image.get_format() == VK_FORMAT_R32G32B32A32_SFLOAT
    buffer = Vk.Buffer(image_size.x * image_size.y * 4 * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, Vk.HostAccess.ReadWrite)

    with ImmediateContext() as cmd:
        cmd.image_barrier(image, Vk.ImageState.General, Vk.ImageState.TransferSrc)
        cmd.copy_image_to_buffer(image, 0, buffer, 0)
        cmd.buffer_barrier(buffer, Vk.BufferState.TransferWrite, Vk.BufferState.HostRead)

    pixels = np.frombuffer(buffer.readback(buffer.get_size()), np.float32)
    return pixels.reshape(image_size.y, image_size.x, 4)


def check_images():
    output_dir = Path("output")
    reference_dir = Path("reference")
    for path in output_dir.glob("*.exr"):
        print(f"checking {path}")
        relative_path = path.relative_to(output_dir)
        reference_path = reference_dir / relative_path
        pixels = load_image(str(path))
        reference = load_image(str(reference_path))
        max_diff = np.max(np.abs(pixels - reference))
        if not np.isclose(max_diff, 0):
            print(f"{path} differs from {reference_path}: max diff = {max_diff:.6f}")
