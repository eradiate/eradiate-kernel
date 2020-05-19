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

.. _sensor-hdistant:

Hemispherical distant directional sensor (:monosp:`hdistant`)
-------------------------------------------------------------

.. pluginparameters::

 * - to_world
   - |transform|
   - Sensor-to-world transformation matrix. Only rotations are allowed.
 * - target
   - |point|
   - *Optional.* Point (in world coordinates) to which sampled rays will be
     cast. Useful for one-dimensional scenes. If unset, rays will be cast
     uniformly over the entire scene.

This sensor plugin implements a distant hemispherical sensor which records
radiation leaving from the scene. By default, this sensor covers the
:math:`Z \geq 0` hemisphere. In this case, the :math:`YZ` plane will lie along
the height of the film, in its middle, while the :math:`ZX` axis will lie along
the width of the film, also in its middle.

.. subfigstart::
.. subfigure:: ../../resources/data-extra/docs/images/sensor/hdistant_mapping.svg
   :caption: Coordinate conventions used when mapping the output image onto the 
             hemisphere with default sensor orientation.
.. subfigend::
    :label: fig-sensor-mapping

An example of XML fragment instantiating this sensor could be:

.. code-block:: xml

    <sensor type="hdistant">
        <vector name="target" value="1, 1, 0"/>
        <transform name="to_world">
            <rotate value="0, 1, 0" angle="45"/>
        </transform>
        <!-- film -->
    </sensor>

*/

MTS_VARIANT class HemisphericalDistantSensor final
    : public Sensor<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Sensor, m_world_transform, m_film)
    MTS_IMPORT_TYPES(Scene)

    HemisphericalDistantSensor(const Properties &props) : Base(props) {
        /* Until `set_scene` is called, we have no information
           about the scene and default to the unit bounding sphere. */
        m_bsphere = ScalarBoundingSphere3f(ScalarPoint3f(0.f), 1.f);

        if (props.has_property("target")) {
            m_target     = props.point3f("target");
            m_has_target = true;
            Log(Debug, "Targeting point %s", m_target);
        } else {
            m_has_target = false;
        }
    }

    void set_scene(const Scene *scene) override {
        m_bsphere = scene->bbox().bounding_sphere();
        m_bsphere.radius =
            max(math::RayEpsilon<Float>,
                m_bsphere.radius * (1.f + math::RayEpsilon<Float>) );
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &direction_sample,
                                          const Point2f &spatial_sample,
                                          Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);
        Ray3f ray;
        ray.time = time;

        // 1. Sample spectrum
        auto [wavelengths, wav_weight] =
            sample_wavelength<Float, Spectrum>(wavelength_sample);
        ray.wavelengths = wavelengths;

        // 2. Sample ray direction
        auto trafo  = m_world_transform->eval(time, active);
        Vector3f v0 = warp::square_to_uniform_hemisphere(spatial_sample);
        ray.d       = trafo.transform_affine(-v0);

        // 3. Sample ray origin
        if (!m_has_target) {
            // If no target point is defined, sample a target point on the
            // bounding sphere's cross section
            Point2f offset =
                warp::square_to_uniform_disk_concentric(direction_sample);
            Vector3f perp_offset = Frame3f(ray.d).to_world(
                Vector3f{ offset.x(), offset.y(), 0.f });
            ray.o = m_bsphere.center + (perp_offset - ray.d) * m_bsphere.radius;
        } else {
            ray.o = m_target - 2.f * ray.d * m_bsphere.radius;
        }

        ray.update();

        return std::make_pair(ray, wav_weight);
    }

    std::pair<RayDifferential3f, Spectrum> sample_ray_differential(
        Float time, Float wavelength_sample, const Point2f &spatial_sample,
        const Point2f &direction_sample, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);
        RayDifferential3f ray;
        ray.time = time;

        // 1. Sample spectrum
        auto [wavelengths, wav_weight] =
            sample_wavelength<Float, Spectrum>(wavelength_sample);
        ray.wavelengths = wavelengths;

        // 2. Sample ray direction
        auto trafo  = m_world_transform->eval(time, active);
        Vector3f v0 = warp::square_to_uniform_hemisphere(spatial_sample);
        ray.d       = trafo.transform_affine(-v0);

        // 3. Sample ray origin
        if (!m_has_target) {
            // If no target point is defined, sample a target point on the
            // bounding sphere's cross section
            Point2f offset =
                warp::square_to_uniform_disk_concentric(direction_sample);
            Vector3f perp_offset = Frame3f(ray.d).to_world(
                Vector3f{ offset.x(), offset.y(), 0.f });
            ray.o = m_bsphere.center + (perp_offset - ray.d) * m_bsphere.radius;
        } else {
            ray.o = m_target - 2.f * ray.d * m_bsphere.radius;
        }

        // 4. Set differentials
        // TODO: compute proper differentials
        ray.d_x               = 0.f;
        ray.d_y               = 0.f;
        ray.has_differentials = false;

        ray.update();

        return std::make_pair(ray, wav_weight);
    }

    /// This sensor does not occupy any particular region of space, return an
    /// invalid bounding box
    ScalarBoundingBox3f bbox() const override { return ScalarBoundingBox3f(); }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "HemisphericalDistantSensor[" << std::endl
            << "  world_transform = " << m_world_transform << "," << std::endl
            << "  bsphere = " << m_bsphere << "," << std::endl
            << "  film = " << m_film << "," << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()

protected:
    ScalarBoundingSphere3f m_bsphere;
    Point3f m_target;
    bool m_has_target;
};

MTS_IMPLEMENT_CLASS_VARIANT(HemisphericalDistantSensor, Sensor)
MTS_EXPORT_PLUGIN(HemisphericalDistantSensor, "HemisphericalDistantSensor");
NAMESPACE_END(mitsuba)
