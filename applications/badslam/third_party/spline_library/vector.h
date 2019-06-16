#pragma once

#include <array>
#include <cmath>

template<size_t dimension, typename floating_t=float>
class Vector
{
public:
    Vector(void) :data() {}
    Vector(std::array<floating_t, dimension> data) :data(data) {}

    inline floating_t& operator[](size_t index) { return data[index]; }
    inline floating_t operator[](size_t index) const { return data[index]; }

    inline Vector<dimension, floating_t> &operator+=(const Vector<dimension, floating_t> &v);
    inline Vector<dimension, floating_t> &operator-=(const Vector<dimension, floating_t> &v);
    inline Vector<dimension, floating_t> &operator*=(floating_t s);
    inline Vector<dimension, floating_t> &operator/=(floating_t s);

    template<size_t d, typename f> friend inline Vector<d, f> operator+(const Vector<d, f> &left, const Vector<d, f> &right);
    template<size_t d, typename f> friend inline Vector<d, f> operator-(const Vector<d, f> &left, const Vector<d, f> &right);
    template<size_t d, typename f> friend inline Vector<d, f> operator*(f s, const Vector<d, f> &v);
    template<size_t d, typename f> friend inline Vector<d, f> operator*(const Vector<d, f> &v, f s);
    template<size_t d, typename f> friend inline Vector<d, f> operator-(const Vector<d, f> &v);
    template<size_t d, typename f> friend inline Vector<d, f> operator/(const Vector<d, f> &v, f s);


    template<size_t d, typename f> friend inline bool operator==(const Vector<d, f> &left, const Vector<d, f> &right);
    template<size_t d, typename f> friend inline bool operator!=(const Vector<d, f> &left, const Vector<d, f> &right);

    inline floating_t length() const;
    inline floating_t lengthSquared() const;

    inline Vector<dimension, floating_t> normalized() const;

    inline static floating_t dotProduct(const Vector<dimension, floating_t>& left, const Vector<dimension, floating_t>& right);

private:
    std::array<floating_t, dimension> data;
};

typedef Vector<2> Vector2;
typedef Vector<3> Vector3;

template<size_t dimension, typename floating_t>
inline Vector<dimension, floating_t> &Vector<dimension, floating_t>::operator+=(const Vector<dimension, floating_t> &other)
{
    for(size_t i = 0; i < dimension; i++) {
        data[i] += other.data[i];
    }
    return *this;
}

template<size_t dimension, typename floating_t>
inline Vector<dimension, floating_t> &Vector<dimension, floating_t>::operator-=(const Vector<dimension, floating_t> &v)
{
    for(size_t i = 0; i < dimension; i++) {
        data[i] -= v.data[i];
    }
    return *this;
}

template<size_t dimension, typename floating_t>
inline Vector<dimension, floating_t> &Vector<dimension, floating_t>::operator*=(floating_t s)
{
    for(size_t i = 0; i < dimension; i++) {
        data[i] *= s;
    }
    return *this;
}

template<size_t dimension, typename floating_t>
inline Vector<dimension, floating_t> &Vector<dimension, floating_t>::operator/=(floating_t s)
{
    for(size_t i = 0; i < dimension; i++) {
        data[i] /= s;
    }
    return *this;
}

template<size_t dimension, typename floating_t>
inline Vector<dimension, floating_t> operator+(const Vector<dimension, floating_t> &left, const Vector<dimension, floating_t> &right)
{
    Vector<dimension, floating_t> result;
    for(size_t i = 0; i < dimension; i++) {
        result.data[i] = left.data[i] + right.data[i];
    }
    return result;
}

template<size_t dimension, typename floating_t>
inline Vector<dimension, floating_t> operator-(const Vector<dimension, floating_t> &left, const Vector<dimension, floating_t> &right)
{
    Vector<dimension, floating_t> result;
    for(size_t i = 0; i < dimension; i++) {
        result.data[i] = left.data[i] - right.data[i];
    }
    return result;
}

template<size_t dimension, typename floating_t>
inline Vector<dimension, floating_t> operator*(floating_t s, const Vector<dimension, floating_t> &v)
{
    Vector<dimension, floating_t> result;
    for(size_t i = 0; i < dimension; i++) {
        result.data[i] = s * v.data[i];
    }
    return result;
}

template<size_t dimension, typename floating_t>
inline Vector<dimension, floating_t> operator*(const Vector<dimension, floating_t> &v, floating_t s)
{
    Vector<dimension, floating_t> result;
    for(size_t i = 0; i < dimension; i++) {
        result.data[i] = v.data[i] * s;
    }
    return result;
}

template<size_t dimension, typename floating_t>
inline Vector<dimension, floating_t> operator-(const Vector<dimension, floating_t> &v)
{
    Vector<dimension, floating_t> result;
    for(size_t i = 0; i < dimension; i++) {
        result.data[i] = -v.data[i];
    }
    return result;
}

template<size_t dimension, typename floating_t>
inline Vector<dimension, floating_t> operator/(const Vector<dimension, floating_t> &v, floating_t s)
{
    Vector<dimension, floating_t> result;
    for(size_t i = 0; i < dimension; i++) {
        result.data[i] = v.data[i] / s;
    }
    return result;
}

template<size_t dimension, typename floating_t>
inline bool operator==(const Vector<dimension, floating_t> &left, const Vector<dimension, floating_t> &right)
{
    for(size_t i = 0; i < dimension; i++) {
        if(left.data[i] != right.data[i])
            return false;
    }
    return true;
}

template<size_t dimension, typename floating_t>
inline bool operator!=(const Vector<dimension, floating_t> &left, const Vector<dimension, floating_t> &right)
{
    for(size_t i = 0; i < dimension; i++) {
        if(left.data[i] != right.data[i])
            return true;
    }
    return false;
}

template<size_t dimension, typename floating_t>
inline Vector<dimension, floating_t> Vector<dimension, floating_t>::normalized() const
{
    floating_t length2 = lengthSquared();
    if (length2 == 0)
        return Vector<dimension, floating_t>();
    else
    {
        floating_t invLength = 1 / std::sqrt(length2);
        return (*this) * invLength;
    }
}

template<size_t dimension, typename floating_t>
inline floating_t Vector<dimension, floating_t>::dotProduct(const Vector<dimension, floating_t>& v1, const Vector<dimension, floating_t>& v2)
{
    floating_t sum(0);
    for(size_t i = 0; i < dimension; i++) {
        sum += v1.data[i] * v2.data[i];
    }
    return sum;
}

template<size_t dimension, typename floating_t>
inline floating_t Vector<dimension, floating_t>::length() const
{
    return std::sqrt(Vector<dimension, floating_t>::dotProduct(*this, *this));
}

template<size_t dimension, typename floating_t>
inline floating_t Vector<dimension, floating_t>::lengthSquared() const
{
    return Vector<dimension, floating_t>::dotProduct(*this, *this);
}
