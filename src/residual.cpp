#include <residual.h>

#ifndef __J_H__
#define __J_H__

#include "comm.h"

#include <Eigen/Dense>

struct searched_line {
    float nx, ny, nz;
    float cx, cy, cz;
    bool ok;
};

template<typename point_type>
inline searched_line search_line(const array_adaptor<point_type>& tree, point_type point) {

    size_t pointSearchInd[5];
    float pointSearchSqDis[5];

    tree.query(point, 5, pointSearchInd, pointSearchSqDis);
    Eigen::Matrix3f A1 = Eigen::Matrix3f::Zero();

    if(pointSearchSqDis[4] < 1.0) {
        float cx = 0, cy = 0, cz = 0;
        for(int j = 0; j < 5; j++) {
            cx += tree.m_data[pointSearchInd[j]].x;
            cy += tree.m_data[pointSearchInd[j]].y;
            cz += tree.m_data[pointSearchInd[j]].z;
        }
        cx /= 5;
        cy /= 5;
        cz /= 5;

        float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
        for(int j = 0; j < 5; j++) {
            float ax = tree.m_data[pointSearchInd[j]].x - cx;
            float ay = tree.m_data[pointSearchInd[j]].y - cy;
            float az = tree.m_data[pointSearchInd[j]].z - cz;

            a11 += ax * ax;
            a12 += ax * ay;
            a13 += ax * az;
            a22 += ay * ay;
            a23 += ay * az;
            a33 += az * az;
        }
        a11 /= 5;
        a12 /= 5;
        a13 /= 5;
        a22 /= 5;
        a23 /= 5;
        a33 /= 5;

        A1(0, 0) = a11;
        A1(0, 1) = a12;
        A1(0, 2) = a13;
        A1(1, 0) = a12;
        A1(1, 1) = a22;
        A1(1, 2) = a23;
        A1(2, 0) = a13;
        A1(2, 1) = a23;
        A1(2, 2) = a33;

        Eigen::EigenSolver<Eigen::Matrix3f> es(A1);
        Eigen::Vector3f D1 = es.eigenvalues().real();
        Eigen::Matrix3f V1 = es.eigenvectors().real();

        if(D1(0) > 3 * D1(1)) {
            searched_line line;
            line.nx = V1(0, 0);
            line.ny = V1(1, 0);
            line.nz = V1(2, 0);
            line.cx = cx;
            line.cy = cy;
            line.cz = cz;
            line.ok = true;
            return line;
        }
    }
    searched_line line;
    line.ok = false;
    return line;
}

struct coeff {
    float px, py, pz;
    float x, y, z;
    float b;
    float s;
};

template<typename point_type>
inline coeff line_coeff(const searched_line& line, const point_type& p) {
    coeff c;
    if(!line.ok) {
        c.px = p.x;
        c.py = p.y;
        c.pz = p.z;
        c.x = 0;
        c.y = 0;
        c.z = 0;
        c.b = 0;
        c.s = 0;
        return c;
    }

    float x0 = p.x;
    float y0 = p.y;
    float z0 = p.z;

    float x1 = line.cx + 0.1 * line.nx;
    float y1 = line.cy + 0.1 * line.ny;
    float z1 = line.cz + 0.1 * line.nz;
    float x2 = line.cx - 0.1 * line.nx;
    float y2 = line.cy - 0.1 * line.ny;
    float z2 = line.cz - 0.1 * line.nz;

    float a012 = sqrt(((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) *
                          ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) +
                      ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) *
                          ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) +
                      ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)) *
                          ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)));

    float l12 = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));

    float la = ((y1 - y2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) +
                (z1 - z2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))) /
        a012 / l12;

    float lb = -((x1 - x2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) -
                 (z1 - z2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) /
        a012 / l12;

    float lc = -((x1 - x2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) +
                 (y1 - y2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) /
        a012 / l12;

    float ld2 = a012 / l12;

    float s = tanf(1 - 0.9 * fabs(ld2));

    c.x = s * la;
    c.y = s * lb;
    c.z = s * lc;
    c.b = s * ld2;
    c.s = s;

    c.px = p.x;
    c.py = p.y;
    c.pz = p.z;

    return c;
}

struct plane {
    float a, b, c, d;
    bool ok;
};

template<typename point_type>
inline plane search_plane(const array_adaptor<point_type>& tree, const point_type& p) {
    size_t pointSearchInd[5];
    float pointSearchSqDis[5];

    tree.query(p, 5, pointSearchInd, pointSearchSqDis);

    Eigen::Matrix<float, 5, 3> matA0;
    Eigen::Matrix<float, 5, 1> matB0;
    Eigen::Vector3f matX0;

    matA0.setZero();
    matB0.fill(-1);
    matX0.setZero();

    if(pointSearchSqDis[4] < 1.0) {
        for(int j = 0; j < 5; j++) {
            matA0(j, 0) = tree.m_data[pointSearchInd[j]].x;
            matA0(j, 1) = tree.m_data[pointSearchInd[j]].y;
            matA0(j, 2) = tree.m_data[pointSearchInd[j]].z;
        }

        matX0 = matA0.colPivHouseholderQr().solve(matB0);

        float pa = matX0(0, 0);
        float pb = matX0(1, 0);
        float pc = matX0(2, 0);
        float pd = 1;

        float ps = sqrt(pa * pa + pb * pb + pc * pc);
        pa /= ps;
        pb /= ps;
        pc /= ps;
        pd /= ps;

        bool planeValid = true;
        for(int j = 0; j < 5; j++) {
            if(fabs(pa * tree.m_data[pointSearchInd[j]].x + pb * tree.m_data[pointSearchInd[j]].y +
                    pc * tree.m_data[pointSearchInd[j]].z + pd) > 0.2) {
                planeValid = false;
                break;
            }
        }

        plane pl;
        pl.a = pa;
        pl.b = pb;
        pl.c = pc;
        pl.d = pd;

        pl.ok = planeValid;
        return pl;
    }

    plane pl;
    pl.ok = false;
    return pl;
}

template<typename point_type>
inline coeff plane_coeff(const plane& pl, const point_type& p) {
    if(pl.ok) {
        float pd2 = pl.a * p.x + pl.b * p.y + pl.c * p.z + pl.d;

        float s = tanf(1 - 0.9 * fabs(pd2) / sqrt(sqrt(p.x * p.x + p.y * p.y + p.z * p.z)));

        coeff c;

        c.x = s * pl.a;
        c.y = s * pl.b;
        c.z = s * pl.c;
        c.b = s * pd2;
        c.s = s;
        c.px = p.x;
        c.py = p.y;
        c.pz = p.z;

        return c;
    }

    coeff c;
    c.s = 0;
    return c;
}

template<typename point_type>
inline point_type search_point(const array_adaptor<point_type>& tree, const point_type& p) {
    size_t pointSearchInd[1];
    float pointSearchSqDis[1];

    tree.query(p, 1, pointSearchInd, pointSearchSqDis);
    return tree.m_data[pointSearchInd[0]];
}

template<typename point_type>
inline coeff point_coeff(const point_type& p1, const point_type& p2) {
    coeff c;
    c.px = p2.x;
    c.py = p2.y;
    c.pz = p2.z;

    auto d = distance2(p1, p2);
    if(d > 0.2f) {
        c.s = 0;
    } else {
        c.s = 1 - d * 5.0f;
    }

    c.x = c.s * (p2.x - p1.x);
    c.y = c.s * (p2.y - p1.y);
    c.z = c.s * (p2.z - p1.z);
    return c;
}

template<typename point_type>
inline void transform_point(point_type& p, const Eigen::Matrix4d& t) {
    Eigen::Vector4d v(p.x, p.y, p.z, 1);
    v = t * v;
    p.x = v.x();
    p.y = v.y();
    p.z = v.z();
}

struct jacobian {
    double j[6];
};

struct jacobian_g {
    double srx, crx, sry, cry, srz, crz;
};

inline void init_jacobian_g(jacobian_g& g, const Transform& t) {
    g.srx = sin(t.roll);
    g.crx = cos(t.roll);
    g.sry = sin(t.pitch);
    g.cry = cos(t.pitch);
    g.srz = sin(t.yaw);
    g.crz = cos(t.yaw);
}

inline jacobian J(const coeff& c, const jacobian_g& g) {
    jacobian j;
    j.j[3] = (c.py * (g.srx * g.srz + g.sry * g.crx * g.crz) +
              c.pz * (g.srz * g.crx - g.srx * g.sry * g.crz)) *
            c.x +
        (c.py * (g.sry * g.srx * g.crx - g.srx * g.crz) -
         c.pz * (g.srx * g.sry * g.srz + g.crx * g.crz)) *
            c.y +
        (c.py * g.crx * g.cry - c.pz * g.srx * g.cry) * c.z;

    j.j[4] = (-c.px * g.sry * g.crz + c.py * g.srx * g.cry * g.crz + c.pz * g.crx * g.cry * g.crz) *
            c.x +
        (-c.px * g.sry * g.srz + c.py * g.srx * g.srz * g.cry + c.pz * g.srz * g.crx * g.cry) *
            c.y +
        (-c.px * g.cry - c.py * g.srx * g.sry - c.pz * g.sry * g.crx) * c.z;

    j.j[5] = (-c.px * g.srz * g.cry + c.py * (-g.sry * g.srx * g.srz - g.crx * g.crz) +
              c.pz * (-g.sry * g.srz * g.crx + g.srx * g.crz)) *
            c.x +
        (c.px * g.cry * g.crz + c.py * (g.sry * g.srx * g.crz - g.srz * g.crx) +
         c.pz * (g.sry * g.crx * g.crz + g.srx * g.srz)) *
            c.y;

    j.j[0] = c.x;
    j.j[1] = c.y;
    j.j[2] = c.z;
    return j;
}

constexpr double NO_VALUE = -125689.123;

static size_t Ab(const feature_objects& source, const feature_adapter& target, const Transform& t,
                 Eigen::MatrixXd& A, Eigen::VectorXd& b, float* _loss) {
    jacobian_g g;
    init_jacobian_g(g, t);
    Eigen::Matrix4d transform = to_eigen(t);

    size_t corner_size = size_of(source.line_features);
    size_t surf_size = size_of(source.plane_features);
    size_t non_size = size_of(source.non_features);

    size_t total_size = corner_size + surf_size + non_size;

    A.resize(total_size, 6);
    b.resize(total_size);

    std::atomic<int> index = 0;
    float loss = 0.0f;
#pragma omp parallel for reduction(+ : loss)
    for(size_t i = 0; i < total_size; ++i) {
        coeff c;
        c.s = 0;
        if(i < corner_size) {
            PointType p2 = source.line_features->at(i);
            transform_point(p2, transform);

            searched_line sl = search_line(target.corner, p2);
            if(sl.ok) {
                c = line_coeff(sl, p2);
            }
        } else if(i < corner_size + surf_size) {
            size_t idx = i - corner_size;
            PointType p2 = source.plane_features->at(idx);
            transform_point(p2, transform);

            plane sp = search_plane(target.surf, p2);
            if(sp.ok) {
                c = plane_coeff(sp, p2);
            }
        } else {
            size_t idx = i - corner_size - surf_size;
            PointType p2 = source.non_features->at(idx);
            transform_point(p2, transform);

            PointType sp = search_point(target.non, p2);
            c = point_coeff(sp, p2);
        }

        if(c.s < 0.1f) {
            continue;
        }

        jacobian j = J(c, g);
        int idx = index.fetch_add(1);
        A(idx, 0) = j.j[0];
        A(idx, 1) = j.j[1];
        A(idx, 2) = j.j[2];
        A(idx, 3) = j.j[3];
        A(idx, 4) = j.j[4];
        A(idx, 5) = j.j[5];
        b(idx) = -c.b;

        loss += c.b * c.b;
    }

    if(_loss != nullptr) {
        *_loss += loss;
    }
    A.conservativeResize(index, 6);
    b.conservativeResize(index);
    return index.load();
}

newton Ab(std::initializer_list<feature_pair> pairs, const Transform& t, float* _loss) {

    std::vector<newton> newtons(pairs.size());

    size_t total_size = 0;
    size_t index = 0;

    if(_loss != nullptr)
        *_loss = 0.0f;

    for(auto&& pair: pairs) {
        total_size += Ab(pair.first, pair.second, t, newtons[index].A, newtons[index].b, _loss);
        index++;
    }

    newton N;
    N.A.resize(total_size, 6);
    N.b.resize(total_size);
    index = 0;

    for(auto&& n: newtons) {
        for(size_t i = 0; i < n.b.rows(); ++i) {
            if(n.b(i) != NO_VALUE) {
                N.A.row(index) = n.A.row(i);
                N.b(index) = n.b(i);
                index++;
            }
        }
    }

    N.top = index;

    if(_loss != nullptr) {
        if(index == 0) {
            *_loss = 10000.0f;
        } else {
            *_loss /= index;
        }
    }

    return N;
}

inline Transform __LM_iteration(const feature_objects& source, array_adaptor<PointType>& corner,
                                array_adaptor<PointType>& surf, array_adaptor<PointType>& non,
                                Eigen::MatrixXd& A, Eigen::VectorXd& b,
                                const Transform& initial_guess, float* loss = nullptr) {

    Eigen::Matrix4d transform = to_eigen(initial_guess);
    jacobian_g g;
    init_jacobian_g(g, initial_guess);

    std::atomic<int> index = 0;

    size_t corner_size = size_of(source.line_features);
    size_t surf_size = size_of(source.plane_features);
    size_t non_size = size_of(source.non_features);

    float __loss = 0.0f;
#pragma omp parallel for(reduction(+ : __loss))
    for(size_t i = 0; i < corner_size + surf_size + non_size; ++i) {
        coeff c;
        c.s = 0;
        if(i < corner_size) {
            PointType p2 = source.line_features->at(i);
            transform_point(p2, transform);

            searched_line sl = search_line(corner, p2);
            if(sl.ok) {
                c = line_coeff(sl, p2);
            }
        } else if(i < corner_size + surf_size) {
            size_t idx = i - corner_size;
            PointType p2 = source.plane_features->at(idx);
            transform_point(p2, transform);

            plane sp = search_plane(surf, p2);
            if(sp.ok) {
                c = plane_coeff(sp, p2);
            }
        } else {
            size_t idx = i - corner_size - surf_size;
            PointType p2 = source.non_features->at(idx);
            transform_point(p2, transform);

            PointType sp = search_point(non, p2);
            c = point_coeff(sp, p2);
        }

        if(c.s < 0.1f) {
            __loss += 1e-3f;
            continue;
        }
        jacobian j = J(c, g);
        int idx = index.fetch_add(1);
        A(idx, 0) = j.j[0];
        A(idx, 1) = j.j[1];
        A(idx, 2) = j.j[2];
        A(idx, 3) = j.j[3];
        A(idx, 4) = j.j[4];
        A(idx, 5) = j.j[5];
        b(idx) = -c.b;

        __loss += c.b * c.b;
    }

    if(index < 100) {
        ROS_INFO("index(%d) < 100, loss set to 10000", index.load());
        loss[0] = 10000.00;
        return initial_guess;
    }

    Eigen::Matrix<double, 6, 6> ATA = A.topRows(index).transpose() * A.topRows(index);
    Eigen::Matrix<double, 6, 1> ATb = A.topRows(index).transpose() * b.topRows(index);
    Eigen::Matrix<double, 6, 1> x = ATA.householderQr().solve(ATb);

    Transform delta;
    delta.x = initial_guess.x + x(0, 0);
    delta.y = initial_guess.y + x(1, 0);
    delta.z = initial_guess.z + x(2, 0);
    delta.roll = initial_guess.roll + x(3, 0);
    delta.pitch = initial_guess.pitch + x(4, 0);
    delta.yaw = initial_guess.yaw + x(5, 0);

    if(loss) {
        loss[0] = __loss / index;
    }
    return delta;
}

Transform LM(const feature_objects& source, const feature_objects& target,
             const Transform& initial_guess, float* loss) {

    size_t corner_size = size_of(source.line_features);
    size_t surf_size = size_of(source.plane_features);
    size_t non_size = size_of(source.non_features);

    size_t total_size = corner_size + surf_size + non_size;

    array_adaptor<PointType> corner(data_of(target.line_features), size_of(target.line_features));
    array_adaptor<PointType> surf(data_of(target.plane_features), size_of(target.plane_features));
    array_adaptor<PointType> non(data_of(target.non_features), size_of(target.non_features));

    Eigen::MatrixXd A(total_size, 6);
    Eigen::VectorXd b(total_size);

    Transform result = initial_guess;
    auto start = std::chrono::high_resolution_clock::now();
    for(int iter = 0; iter < 30; ++iter) {
        Transform u = __LM_iteration(source, corner, surf, non, A, b, result, loss);
        float deltaR =
            sqrtf(p2(u.roll - result.roll) + p2(u.pitch - result.pitch) + p2(u.yaw - result.yaw));
        float deltaT = sqrtf(p2(u.x - result.x) + p2(u.y - result.y) + p2(u.z - result.z));
        // printf("iter: %d, deltaR: %f, deltaT: %f\r\n", iter, deltaR, deltaT);
        result = u;
        if(deltaR < 0.0005 && deltaT < 0.0005) {
            result.roll = result.pitch = 0;
            break;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    return result;
}

#endif
