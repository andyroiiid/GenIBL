// Copyright 2026 Andrew Huang. All Rights Reserved.

static const float PI = 3.141592653589793;
static const float TWO_PI = 2.0 * PI;

float3x3 UV2TBN(float2 uv)
{
    const float yaw = uv.x * TWO_PI;
    const float pitch = (uv.y - 0.5f) * PI;
    const float cosYaw = cos(yaw);
    const float sinYaw = sin(yaw);
    const float cosPitch = cos(pitch);
    const float sinPitch = sin(pitch);
    return float3x3(
        -sinYaw, 0.0f, cosYaw,
        -sinPitch * cosYaw, cosPitch, -sinPitch * sinYaw,
        cosPitch * cosYaw, sinPitch, cosPitch * sinYaw
    );
}

float2 Sphere2UV(float3 v)
{
    const float yaw = atan2(v.z, v.x);
    const float pitch = asin(v.y);
    return float2(
        frac(yaw / TWO_PI + 1.0f),
        pitch / PI + 0.5f
    );
}

// R2 sequence
float2 R2(uint i)
{
    return frac(0.5f + i * float2(0.7548776662466927f, 0.5698402909980532f));
}

float3 ImportanceSampleGGX(float2 xi, float roughness)
{
    const float a = roughness * roughness;
    const float a2 = a * a;

    const float phi = TWO_PI * xi.x;
    const float cosTheta = sqrt((1.0f - xi.y) / (1.0f + (a2 - 1.0f) * xi.y));
    const float sinTheta = sqrt(1.0f - cosTheta * cosTheta);

    return float3(
        sinTheta * cos(phi),
        sinTheta * sin(phi),
        cosTheta
    );
}

// The uint32_t version of DescriptorHandle<T>
struct BindlessHandle<T> : IComparable where T : IOpaqueDescriptor
{
    uint32_t handle;

    [ForceInline]
    __init(uint32_t handle) {
        this.handle = handle;
    }

    [ForceInline]
    DescriptorHandle<T> operator()() {
        return DescriptorHandle<T>(handle);
    }

    [ForceInline]
    bool equals(BindlessHandle<T> other)
    {
        return this.handle == other.handle;
    }

    [ForceInline]
    bool lessThan(BindlessHandle<T> other)
    {
        return this.handle < other.handle;
    }

    [ForceInline]
    bool lessThanOrEquals(BindlessHandle<T> other)
    {
        return this.handle <= other.handle;
    }
};
