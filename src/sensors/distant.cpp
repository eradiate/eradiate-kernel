#include <mitsuba/core/bbox.h>
#include <mitsuba/core/bsphere.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/sensor.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _sensor-distant:

Distant directional sensor (:monosp:`distant`)
----------------------------------------------

.. pluginparameters::

 * - to_world
   - |transform|
   - Emitter-to-world transformation matrix.
 * - direction
   - |vector|
   - Alternative (and exclusive) to `to_world`. Direction from which the
     sensor will be recording in world coordinates.
 * - target
   - |point|
   - *Optional.* Point (in world coordinates) to which sampled rays will be
     shot. Useful for one-dimensional scenes.

This sensor plugin implements a distant directional sensor, which records
radiation emitted in a fixed direction. If the ``target`` parameter is not set,
the rays will be distributed over the :math:`Z=0`-plane of the global bounding
box.

*/

MTS_VARIANT class DistantSensor final : public Sensor<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Sensor, m_world_transform, m_needs_sample_3, m_film,
                    m_sampler, m_resolution, m_shutter_open,
                    m_shutter_open_time, m_aspect)
    MTS_IMPORT_TYPES(Scene, Texture)

    DistantSensor(const Properties &props) : Base(props) {
        /* Until `set_scene` is called, we have no information
           about the scene and default to the unit bounding sphere. */
        m_bsphere = ScalarBoundingSphere3f(ScalarPoint3f(0.f), 1.f);

        if (props.has_property("direction")) {
            if (props.has_property("to_world"))
                Throw("Only one of the parameters 'direction' and 'to_world' "
                      "can be specified at the same time!'");

            ScalarVector3f direction(normalize(props.vector3f("direction")));
            auto [up, unused] = coordinate_system(direction);

            m_world_transform =
                new AnimatedTransform(ScalarTransform4f::look_at(
                    ScalarPoint3f(0.0f), ScalarPoint3f(direction), up));
        }

        if (props.has_property("target")) {
            m_target     = props.point3f("target");
            m_has_target = true;
            Log(Debug, "Targeting point %s", m_target);
        } else {
            m_has_target = false;
        }

        if (m_film->size() != ScalarPoint2i(1, 1))
            Throw("This sensor only supports films of size 1x1 Pixels!");

        if (m_film->reconstruction_filter()->radius() >
            0.5f + math::RayEpsilon<Float>)
            Log(Warn, "This sensor should be used with a reconstruction filter "
                      "with a radius of 0.5 or lower (e.g. default box)");
    }

    void set_scene(const Scene *scene) override {
        m_bsphere = scene->bbox().bounding_sphere();
        m_bsphere.radius =
            max(math::RayEpsilon<Float>,
                m_bsphere.radius * (1.f + math::RayEpsilon<Float>));
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &spatial_sample,
                                          const Point2f & /*direction_sample*/,
                                          Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);
        Ray3f ray;
        ray.time = time;

        // 1. Sample spectrum
        auto [wavelengths, weight] =
            sample_wavelength<Float, Spectrum>(wavelength_sample);
        ray.wavelengths = wavelengths;

        // 2. Set ray direction
        auto trafo = m_world_transform->eval(time, active);
        ray.d      = trafo.transform_affine(Vector3f{ 0.f, 0.f, 1.f });

        // 3. Sample ray origin
        if (!m_has_target) {
            // If no target point is defined, sample a target point on the
            // bounding sphere's cross section
            Point2f p = warp::square_to_uniform_disk_concentric(spatial_sample);
            Vector3f perp_offset = trafo.transform_affine(
                Vector3f{ p.x(), p.y(), 0.f } * m_bsphere.radius);
            ray.o = m_bsphere.center - ray.d * m_bsphere.radius + perp_offset;
        } else {
            ray.o = m_target - 2. * ray.d * m_bsphere.radius;
        }

        ray.update();

        return std::make_pair(
            ray, unpolarized<Spectrum>(weight) *
                     (4.f * sqr(math::Pi<Float> * m_bsphere.radius)));
    }

    std::pair<RayDifferential3f, Spectrum> sample_ray_differential(
        Float time, Float wavelength_sample, const Point2f &spatial_sample,
        const Point2f & /*aperture_sample*/, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);
        RayDifferential3f ray;
        ray.time = time;

        // 1. Sample spectrum
        auto [wavelengths, weight] =
            sample_wavelength<Float, Spectrum>(wavelength_sample);
        ray.wavelengths = wavelengths;

        // 2. Set ray direction
        auto trafo = m_world_transform->eval(time, active);
        ray.d      = trafo.transform_affine(Vector3f{ 0.f, 0.f, 1.f });

        // 3. Sample ray origin
        if (!m_has_target) {
            // If no target point is defined, sample a target point on the
            // bounding sphere's cross section
            Point2f p = warp::square_to_uniform_disk_concentric(spatial_sample);
            Vector3f perp_offset = trafo.transform_affine(
                Vector3f{ p.x(), p.y(), 0.f } * m_bsphere.radius);
            ray.o = m_bsphere.center - ray.d * m_bsphere.radius + perp_offset;
        } else {
            ray.o = m_target - 2. * ray.d * m_bsphere.radius;
        }

        // 4. Set differentials; since the film size is always 1x1, we don't
        //    have differentials
        ray.has_differentials = false;

        return std::make_pair(
            ray, unpolarized<Spectrum>(weight) *
                     (4.f * sqr(math::Pi<Float> * m_bsphere.radius)));
    }

    /// This emitter does not occupy any particular region of space, return an
    /// invalid bounding box
    ScalarBoundingBox3f bbox() const override { return ScalarBoundingBox3f(); }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "DistantSensor[" << std::endl
            << "  world_transform = " << m_world_transform << "," << std::endl
            << "  bsphere = " << m_bsphere << "," << std::endl
            << "  film = " << m_film << "," << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()

protected:
    ScalarBoundingSphere3f m_bsphere;
    ScalarPoint3f m_target;
    bool m_has_target;
};

MTS_IMPLEMENT_CLASS_VARIANT(DistantSensor, Sensor)
MTS_EXPORT_PLUGIN(DistantSensor, "DistantSensor");
NAMESPACE_END(mitsuba)
