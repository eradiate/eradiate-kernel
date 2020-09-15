#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/shape.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _shape-cone:

Cone (:monosp:`cone`)
----------------------------------------------------

.. pluginparameters::


 * - p0
   - |point|
   - Object-space starting point of the cone's centerline. (Base)
     (Default: (0, 0, 0))
 * - p1
   - |point|
   - Object-space endpoint of the cone's centerline (Default: (0, 0, 1)) (Tip)
 * - radius
   - |float|
   - Radius of the cone in object-space units (Default: 1)
 * - flip_normals
   - |bool|
   -  Is the cone inverted, i.e. should the normal vectors
      be flipped? (Default: |false|, i.e. the normals point outside)
 * - to_world
   - |transform|
   - Specifies an optional linear object-to-world transformation. Note that non-uniform scales are
     not permitted! (Default: none, i.e. object space = world space)

.. subfigstart::
.. subfigure:: ../../resources/data-extra/docs/images/render/shape_cone_onesided.jpg
   :caption: Cone with the default configuration and diffuse BSDF
.. subfigure:: ../../resources/data-extra/docs/images/render/shape_cone_twosided.jpg
   :caption: Upside down cone with two-sided shading
.. subfigend::
   :label: fig-cone

This shape plugin describes a simple cone intersection primitive.
It should always be preferred over approximations modeled using
triangles. Note that the cone does not have endcaps -- also,
its normals point outward, which means that the inside will be treated
as fully absorbing by most material models. If this is not
desirable, consider using the :ref:`twosided <bsdf-twosided>` plugin.

A simple example for instantiating a cone, whose interior is visible:

.. code-block:: xml

    <shape type="cone">
        <float name="radius" value="0.3"/>
        <bsdf type="twosided">
            <bsdf type="diffuse"/>
        </bsdf>
    </shape>
 */

template <typename Float, typename Spectrum>
class Cone final : public Shape<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Shape, m_to_world, m_to_object, set_children,
                    get_children_string, parameters_grad_enabled)
    MTS_IMPORT_TYPES()

    using typename Base::ScalarIndex;
    using typename Base::ScalarSize;

    Cone(const Properties &props) : Base(props) {
        /// Are the surface normals pointing inwards? default: no
        m_flip_normals = props.bool_("flip_normals", false);

        update();
        set_children();
    }

    void update() {
         // Extract center and radius from to_world matrix (25 iterations for numerical accuracy)
        auto [S, Q, T] = transform_decompose(m_to_world.matrix, 25);

        if (abs(S[0][1]) > 1e-6f || abs(S[0][2]) > 1e-6f || abs(S[1][0]) > 1e-6f ||
            abs(S[1][2]) > 1e-6f || abs(S[2][0]) > 1e-6f || abs(S[2][1]) > 1e-6f)
            Log(Warn, "'to_world' transform shouldn't contain any shearing!");

        if (!(abs(S[0][0] - S[1][1]) < 1e-6f))
            Log(Warn, "'to_world' transform shouldn't contain non-uniform scaling along the X and Y axes!");

        m_radius = S[0][0];
        m_length = S[2][2];

        if (m_radius <= 0.f) {
            m_radius = std::abs(m_radius);
            m_flip_normals = !m_flip_normals;
        }

        // Reconstruct the to_world transform with uniform scaling and no shear
        m_to_world = transform_compose(ScalarMatrix3f(1.f), Q, T);
        m_to_object = m_to_world.inverse();

        m_inv_surface_area = rcp(surface_area());
    }

    ScalarBoundingBox3f bbox() const override {
        ScalarVector3f x1 = m_to_world * ScalarVector3f(m_radius, 0.f, 0.f),
                       x2 = m_to_world * ScalarVector3f(0.f, m_radius, 0.f),
                       x  = sqrt(sqr(x1) + sqr(x2));

        ScalarPoint3f p0 = m_to_world * ScalarPoint3f(0.f, 0.f, 0.f),
                      p1 = m_to_world * ScalarPoint3f(0.f, 0.f, m_length);

        /* To bound the cone, it is sufficient to find the
           smallest box containing the two circles at the endpoints. */
        return ScalarBoundingBox3f(min(p0 - x, p1 - x), max(p0 + x, p1 + x));
    }

    ScalarBoundingBox3f bbox(ScalarIndex /*index*/, const ScalarBoundingBox3f &clip) const override {
        using FloatP8         = Packet<ScalarFloat, 8>;
        using MaskP8          = mask_t<FloatP8>;
        using Point3fP8       = Point<FloatP8, 3>;
        using Vector3fP8      = Vector<FloatP8, 3>;
        using BoundingBox3fP8 = BoundingBox<Point3fP8>;

        ScalarPoint3f cyl_p = m_to_world.transform_affine(ScalarPoint3f(0.f, 0.f, 0.f));
        ScalarVector3f cyl_d =
            m_to_world.transform_affine(ScalarVector3f(0.f, 0.f, m_length));

        // Compute a base bounding box
        ScalarBoundingBox3f bbox(this->bbox());
        bbox.clip(clip);

        /* Now forget about the cone ends and intersect an infinite
           cylinder with each bounding box face, then compute a bounding
           box of the resulting ellipses. */
        Point3fP8 face_p = zero<Point3fP8>();
        Vector3fP8 face_n = zero<Vector3fP8>();

        for (size_t i = 0; i < 3; ++i) {
            face_p.coeff(i,  i * 2 + 0) = bbox.min.coeff(i);
            face_p.coeff(i,  i * 2 + 1) = bbox.max.coeff(i);
            face_n.coeff(i,  i * 2 + 0) = -1.f;
            face_n.coeff(i,  i * 2 + 1) = 1.f;
        }

        // Project the cone direction onto the plane
        FloatP8 dp   = dot(cyl_d, face_n);
        MaskP8 valid = neq(dp, 0.f);

        // Compute semimajor/minor axes of ellipse
        Vector3fP8 v1 = fnmadd(face_n, dp, cyl_d);
        FloatP8 v1_n2 = squared_norm(v1);
        v1 = select(neq(v1_n2, 0.f), v1 * rsqrt(v1_n2),
                    coordinate_system(face_n).first);
        Vector3fP8 v2 = cross(face_n, v1);

        // Compute length of axes
        v1 *= m_radius / abs(dp);
        v2 *= m_radius;

        // Compute center of ellipse
        FloatP8 t = dot(face_n, face_p - cyl_p) / dp;
        Point3fP8 center = fmadd(Vector3fP8(cyl_d), t, Vector3fP8(cyl_p));
        center[neq(face_n, 0.f)] = face_p;

        // Compute ellipse minima and maxima
        Vector3fP8 x = sqrt(sqr(v1) + sqr(v2));
        BoundingBox3fP8 ellipse_bounds(center - x, center + x);
        MaskP8 ellipse_overlap = valid && bbox.overlaps(ellipse_bounds);
        ellipse_bounds.clip(bbox);

        return ScalarBoundingBox3f(
            hmin_inner(select(ellipse_overlap, ellipse_bounds.min,
                              Point3fP8(std::numeric_limits<ScalarFloat>::infinity()))),
            hmax_inner(select(ellipse_overlap, ellipse_bounds.max,
                              Point3fP8(-std::numeric_limits<ScalarFloat>::infinity()))));
    }

    ScalarFloat surface_area() const override {
        ScalarFloat s = sqrt(sqr(m_radius) + sqr(m_length));
        return math::Pi<ScalarFloat> * m_radius * s;
    }

    // =============================================================
    //! @{ \name Sampling routines
    // =============================================================

    PositionSample3f sample_position(Float time, const Point2f &sample,
                                     Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        Float phi = math::TwoPi<Float> * sample.y();
        auto [sinphi, cosphi] = sincos(phi);

        Float x = m_radius*(sample.x())*cosphi,
              y = m_radius*(sample.x())*sinphi,
              z = m_length*(1-sample.x());

        Point3f p = Point3f(x, y, z);

        // p in cone coordinates
        Point3f p_cc = Point3f(sqrt(sqr(p.x()) + sqr(p.y()))/p.z(),
                                      atan2(p.y(), p.x()),
                                      p.z());

        auto [sin_phi, cos_phi] = sincos(p_cc.y());

        Normal3f n(cos_phi, sin_phi, -p_cc.x());

        n = n * p_cc.x() * p_cc.z();


        if (m_flip_normals)
            n *= -1;

        PositionSample3f ps;
        ps.p     = m_to_world.transform_affine(p);
        ps.n     = normalize(n);
        ps.pdf   = m_inv_surface_area;
        ps.time  = time;
        ps.delta = false;
        return ps;
    }

    Float pdf_position(const PositionSample3f & /*ps*/, Mask active) const override {
        MTS_MASK_ARGUMENT(active);
        return m_inv_surface_area;
    }

    //! @}
    // =============================================================

    // =============================================================
    //! @{ \name Ray tracing routines
    // =============================================================

    PreliminaryIntersection3f ray_intersect_preliminary(const Ray3f &ray_,
                                                        Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        using Double = std::conditional_t<is_cuda_array_v<Float>, Float, Float64>;

        Ray3f ray = m_to_object.transform_affine(ray_);
        Double mint = Double(ray.mint),
               maxt = Double(ray.maxt);

        Double ox = Double(ray.o.x()), oy = Double(ray.o.y()),
              oz = Double(ray.o.z()), dx = Double(ray.d.x()),
              dy = Double(ray.d.y()), dz = Double(ray.d.z());

        scalar_t<Double> radius = scalar_t<Double>(m_radius),
                        length = scalar_t<Double>(m_length),
                        k      = radius / length;
        k *= k;

        Double A = sqr(dx) + sqr(dy) - k * sqr(dz),
               B = 2.f * (dx * ox + dy * oy - k * dz * (oz - length)),
               C = sqr(ox) + sqr(oy) - k * sqr(oz - length);

        auto [solution_found, near_t, far_t] = math::solve_quadratic(A, B, C);

        // Cone doesn't intersect with the segment on the ray
        Mask out_bounds = !(near_t <= maxt && far_t >= mint); // NaN-aware conditionals

        Double z_pos_near = oz + dz*near_t,
               z_pos_far  = oz + dz*far_t;

        // Cone fully contains the segment of the ray
        Mask in_bounds = near_t < mint && far_t > maxt;

        Mask shadow_cone_near = (length - (oz + (dz * near_t)) <= 0),
             shadow_cone_far  = (length - (oz + (dz * far_t)) <= 0);

        active &= solution_found && !out_bounds && !in_bounds && 
                  ((z_pos_near >= 0 && near_t >= mint && !shadow_cone_near) ||
                   (z_pos_far  >= 0 && far_t <= maxt && !shadow_cone_far));

        PreliminaryIntersection3f pi = zero<PreliminaryIntersection3f>();
        pi.t = select(active,
                      select(z_pos_near >= 0 && near_t >= mint && !shadow_cone_near,
                             Float(near_t), select(!shadow_cone_far, Float(far_t), math::Infinity<Float>)),
                      math::Infinity<Float>);
        pi.shape = this;

        return pi;
    }

    Mask ray_test(const Ray3f &ray_, Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        using Double = std::conditional_t<is_cuda_array_v<Float>, Float, Float64>;

        Ray3f ray = m_to_object.transform_affine(ray_);
        Double mint = Double(ray.mint);
        Double maxt = Double(ray.maxt);

        Double ox = Double(ray.o.x()), oy = Double(ray.o.y()),
               oz = Double(ray.o.z()), dx = Double(ray.d.x()),
               dy = Double(ray.d.y()), dz = Double(ray.d.z());

        scalar_t<Double> radius    = scalar_t<Double>(m_radius),
                         length    = scalar_t<Double>(m_length),
                         k         = radius / length;
        k *= k;

        Double A = sqr(dx) + sqr(dy) - k * sqr(dz),
               B = 2.f * (dx * ox + dy * oy - k * dz * (oz - length)),
               C = sqr(ox) + sqr(oy) - k * sqr(oz - length);

        auto [solution_found, near_t, far_t] = math::solve_quadratic(A, B, C);

        // Cone doesn't intersect with the segment on the ray
        Mask out_bounds = !(near_t <= maxt && far_t >= mint); // NaN-aware conditionals

        Double z_pos_near = oz + dz*near_t,
               z_pos_far  = oz + dz*far_t;

        // Cone fully contains the segment of the ray
        Mask in_bounds = near_t < mint && far_t > maxt;

        Mask shadow_cone_near = (length - (oz + (dz * near_t)) <= 0),
             shadow_cone_far  = (length - (oz + (dz * far_t)) <= 0);

        Mask valid_intersection =
            active && solution_found && !out_bounds && !in_bounds && 
            ((z_pos_near >= 0 && near_t >= mint && !shadow_cone_near) ||
             (z_pos_far >= 0 && far_t <= maxt && !shadow_cone_far));

        return valid_intersection;
    }

    SurfaceInteraction3f compute_surface_interaction(const Ray3f &ray,
                                                     PreliminaryIntersection3f pi,
                                                     HitComputeFlags flags,
                                                     Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        bool differentiable = false;
        if constexpr (is_diff_array_v<Float>)
            differentiable = requires_gradient(ray.o) ||
                             requires_gradient(ray.d) ||
                             parameters_grad_enabled();

        // Recompute ray intersection to get differentiable prim_uv and t
        if (differentiable && !has_flag(flags, HitComputeFlags::NonDifferentiable))
            pi = ray_intersect_preliminary(ray, active);

        active &= pi.is_valid();

        SurfaceInteraction3f si = zero<SurfaceInteraction3f>();
        si.t = select(active, pi.t, math::Infinity<Float>);

        si.p = ray(pi.t);

        Vector3f local = m_to_object.transform_affine(si.p);

        Float phi = atan2(local.y(), local.x());
        masked(phi, phi < 0.f) += 2.f * math::Pi<Float>;

        si.uv = Point2f(phi * math::InvTwoPi<Float>, local.z() / m_length);

        Vector3f dp_du = 2.f * math::Pi<Float> * Vector3f(-local.y(), local.x(), 0.f);
        Vector3f dp_dv = Vector3f(-local.x()/(1.f - si.uv.y()), 
                                  -local.y()/(1.f - si.uv.y()), 
                                  m_length);
        si.dp_du = m_to_world.transform_affine(dp_du);
        si.dp_dv = m_to_world.transform_affine(dp_dv);
        si.n = Normal3f(normalize(cross(si.dp_du, si.dp_dv)));

        if (m_flip_normals)
            si.n *= -1.f;

        /* Mitigate roundoff error issues by a normal shift of the computed
           intersection point */
        si.p += si.n * ((m_radius * (m_length-local.z())/m_length) - norm(head<2>(local)));

        si.sh_frame.n = si.n;
        si.time = ray.time;

        if (has_flag(flags, HitComputeFlags::dNSdUV)) {
            // Compute cone $\dndu$ and $\dndv$
            Vector3f d2Pduu = -1*sqr(math::TwoPi<Float>) * Vector3f(local.x(), local.y(), 0.);
            Vector3f d2Pduv = math::TwoPi<Float> / (1.f - si.uv.y()) * Vector3f(local.y(), -local.x(), 0.);
            Vector3f d2Pdvv(0, 0, 0);

            // Compute coefficients for fundamental forms
            Float E = dot(si.dp_du, si.dp_du);
            Float F = dot(si.dp_du, si.dp_dv);
            Float G = dot(si.dp_du, si.dp_du);
            Float e = dot(si.n, d2Pduu);
            Float f = dot(si.n, d2Pduv);
            Float g = dot(si.n, d2Pdvv);

            // Compute $\dndu$ and $\dndv$ from fundamental form coefficients
            Float invEGF2 = 1 / (E * G - F * F);

            si.dn_du = Normal3f((f * F - e * G) * invEGF2 * si.dp_du + (e * F - f * E) * invEGF2 * si.dp_dv);
            si.dn_dv = Normal3f((g * F - f * G) * invEGF2 * si.dp_du + (f * F - g * E) * invEGF2 * si.dp_dv);
        }

        return si;
    }

    //! @}
    // =============================================================

    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
    }

    void parameters_changed(const std::vector<std::string> &/*keys*/) override {
        update();
        Base::parameters_changed();
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "Cone[" << std::endl
            << "  to_world = " << string::indent(m_to_world, 13) << "," << std::endl
            << "  radius = "  << m_radius << "," << std::endl
            << "  length = "  << m_length << "," << std::endl
            << "  surface_area = " << surface_area() << "," << std::endl
            << "  " << string::indent(get_children_string()) << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ScalarFloat m_radius, m_length;
    ScalarFloat m_inv_surface_area;
    bool m_flip_normals;
};

MTS_IMPLEMENT_CLASS_VARIANT(Cone, Shape)
MTS_EXPORT_PLUGIN(Cone, "Cone intersection primitive");
NAMESPACE_END(mitsuba)
