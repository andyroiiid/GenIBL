# Copyright 2026 Andrew Huang. All Rights Reserved.

from pathlib import Path

import OpenImageIO as oiio

output_dir = Path("output")

radiance_files = sorted(output_dir.glob("radiance_*.exr"))
radiance_images = [oiio.ImageInput.open(str(filename)) for filename in radiance_files]
open_modes = ["Create" if i == 0 else "AppendMIPLevel" for i in range(len(radiance_images))]

out = oiio.ImageOutput.create("radiance.exr")

for file, mip, open_mode in zip(radiance_files, radiance_images, open_modes):
    print(f"Appending {file}")
    spec = mip.spec()
    spec.attribute("textureformat", "Plain Texture")
    spec.attribute("openexr:levelmode", 1)  # EXR_TILE_MIPMAP_LEVELS
    spec.tile_width = 64
    spec.tile_height = 64
    out.open("radiance.exr", spec, open_mode)
    pixels = mip.read_image()
    out.write_image(pixels)

out.close()
