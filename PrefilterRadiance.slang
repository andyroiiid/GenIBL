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
    float uRoughness;
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
    const float3 N = mul(float3(0.0f, 0.0f, 1.0f), tbn);
    const float3 V = N;

    const uint NUM_SAMPLES = 10000;
    float3 color = 0.0f;
    float totalWeight = 0.0f;
    for (uint i = 0; i < NUM_SAMPLES; i++)
    {
        const float2 xi = R2(i);
        const float3 H = mul(ImportanceSampleGGX(xi, uRoughness), tbn);
        const float3 L = normalize(2.0f * dot(V, H) * H - V);

        const float NdotL = saturate(dot(N, L));
        if (NdotL > 0.0f)
        {
            const float2 sampleUv = Sphere2UV(L);
            color += uInput().SampleLevel(uSampler(), sampleUv, 0).rgb * NdotL;
            totalWeight += NdotL;
        }
    }
    uOutput()[pos] = float4(color / totalWeight, 1.0f);
}
