// Copyright 2026 Andrew Huang. All Rights Reserved.

import Common;

[push_constant]
cbuffer PushConstants
{
    BindlessHandle<SamplerState> uSampler;
    BindlessHandle<Texture2D<float4>> uInput;
    BindlessHandle<RWTexture2D<float4>> uOutput;
    uint2 uSize;
    uint uPosY;
};

[shader("compute")]
[numthreads(32, 1, 1)]
void MainCS(const uint dtid : SV_DispatchThreadID)
{
    const uint2 pos = uint2(dtid, uPosY);
    if (pos.x >= uSize.x || pos.y >= uSize.y)
    {
        return;
    }

    const float2 uv = float2(pos) / float2(uSize);
    const float3x3 tbn = UV2TBN(uv);

    const uint NUM_SAMPLES = 10000;
    float4 irradiance = 0.0f;
    for (uint i = 0; i < NUM_SAMPLES; i++)
    {
        const float2 xi = R2(i);

        const float phi = TWO_PI * xi.x;
        const float cosPhi = cos(phi);
        const float sinPhi = sin(phi);

        const float cosTheta = sqrt(1.0f - xi.y);
        const float sinTheta = sqrt(xi.y);

        const float3 tangentSpaceVec = float3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
        const float3 worldSpaceVec = mul(tangentSpaceVec, tbn);
        const float2 sampleUv = Sphere2UV(worldSpaceVec);
        irradiance += uInput().SampleLevel(uSampler(), sampleUv, 0);
    }
    uOutput()[pos] = irradiance / NUM_SAMPLES;
}
