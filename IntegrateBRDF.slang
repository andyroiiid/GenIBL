// Copyright 2026 Andrew Huang. All Rights Reserved.

import Common;

[push_constant]
cbuffer PushConstants
{
    BindlessHandle<RWTexture2D<float4>> uOutput;
    uint2 uSize;
    uint uPosY;
};

// GGX/Schlick-Beckmann
float GeometrySchlickGGX(const float NdotV, const float k)
{
    return NdotV / lerp(k, 1.0f, NdotV);
}

// Smith's method
float GeometrySmith(const float NdotL, const float NdotV, const float k)
{
    const float lightAttenuation = GeometrySchlickGGX(NdotL, k);
    const float viewAttenuation = GeometrySchlickGGX(NdotV, k);
    return lightAttenuation * viewAttenuation;
}

float GeometrySmithIBL(const float NdotL, const float NdotV, const float roughness)
{
    const float k = roughness * roughness * 0.5f;
    return GeometrySmith(NdotL, NdotV, k);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void MainCS(const uint dtid : SV_DispatchThreadID)
{
    const uint2 pos = uint2(dtid, uPosY);
    if (pos.x >= uSize.x || pos.y >= uSize.y)
    {
        return;
    }

    const float2 uv = float2(pos) / float2(uSize - 1);
    const float NdotV = max(uv.x, 0.001f);
    const float roughness = max(uv.y, 0.001f);

    const float3 V = float3(sqrt(1.0f - NdotV * NdotV), 0.0f, NdotV);
    const float3 N = float3(0.0f, 0.0f, 1.0f);

    const uint NUM_SAMPLES = 10000;
    float scale = 0.0f;
    float bias = 0.0f;
    for (uint i = 0; i < NUM_SAMPLES; i++)
    {
        const float2 xi = R2(i);
        const float3 H = ImportanceSampleGGX(xi, roughness);
        const float3 L = -reflect(V, H);

        const float NdotL = saturate(L.z);
        const float NdotH = saturate(H.z);
        const float VdotH = saturate(dot(V, H));

        if (NdotL > 0.0f)
        {
            const float G = GeometrySmithIBL(NdotL, NdotV, roughness);
            const float G_Vis = G * VdotH / (NdotH * NdotV);
            const float Fc = pow(1.0f - VdotH, 5.0f);

            scale += G_Vis * (1.0f - Fc);
            bias += G_Vis * Fc;
        }
    }
    uOutput()[pos] = float4(scale / NUM_SAMPLES, bias / NUM_SAMPLES, 0.0f, 1.0f);
}
