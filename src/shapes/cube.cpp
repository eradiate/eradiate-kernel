#include <mitsuba/core/fwd.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/mesh.h>
#include <mitsuba/render/sensor.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _shape-cube:

Cube (:monosp:`cube`)
-------------------------

.. pluginparameters::

 * - to_world
   - |transform|
   - Specifies an optional linear object-to-world transformation.
     (Default: none (i.e. object space = world space))

This shape plugin describes a cube intersection primitive, based on the triangle
mesh class.

*/

MTS_VARIANT class Cube final : public Mesh<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Mesh, m_name, m_bbox, m_to_world, m_vertex_count,
                    m_face_count, m_vertex_positions_buf, m_vertex_normals_buf,
                    m_vertex_texcoords_buf, m_faces_buf, m_mesh_attributes,
                    m_disable_vertex_normals, has_vertex_normals,
                    has_vertex_texcoords, recompute_vertex_normals,
                    set_children)
    MTS_IMPORT_TYPES()

    using typename Base::FloatStorage;
    using typename Base::InputFloat;
    using typename Base::InputNormal3f;
    using typename Base::InputPoint3f;
    using typename Base::InputVector2f;
    using typename Base::InputVector3f;
    using typename Base::ScalarIndex;
    using typename Base::ScalarSize;
    using ScalarIndex3 = std::array<ScalarIndex, 3>;

private:
    const std::vector<InputVector3f> vertices = {
        { 1, -1, -1 }, { 1, -1, 1 },  { -1, -1, 1 },  { -1, -1, -1 },
        { 1, 1, -1 },  { -1, 1, -1 }, { -1, 1, 1 },   { 1, 1, 1 },
        { 1, -1, -1 }, { 1, 1, -1 },  { 1, 1, 1 },    { 1, -1, 1 },
        { 1, -1, 1 },  { 1, 1, 1 },   { -1, 1, 1 },   { -1, -1, 1 },
        { -1, -1, 1 }, { -1, 1, 1 },  { -1, 1, -1 },  { -1, -1, -1 },
        { 1, 1, -1 },  { 1, -1, -1 }, { -1, -1, -1 }, { -1, 1, -1 }
    };
    const std::vector<InputNormal3f> normals = {
        { 0, -1, 0 }, { 0, -1, 0 }, { 0, -1, 0 }, { 0, -1, 0 }, { 0, 1, 0 },
        { 0, 1, 0 },  { 0, 1, 0 },  { 0, 1, 0 },  { 1, 0, 0 },  { 1, 0, 0 },
        { 1, 0, 0 },  { 1, 0, 0 },  { 0, 0, 1 },  { 0, 0, 1 },  { 0, 0, 1 },
        { 0, 0, 1 },  { -1, 0, 0 }, { -1, 0, 0 }, { -1, 0, 0 }, { -1, 0, 0 },
        { 0, 0, -1 }, { 0, 0, -1 }, { 0, 0, -1 }, { 0, 0, -1 }
    };
    const std::vector<InputVector2f> texcoords = {
        { 0, 1 }, { 1, 1 }, { 1, 0 }, { 0, 0 }, { 0, 1 }, { 1, 1 },
        { 1, 0 }, { 0, 0 }, { 0, 1 }, { 1, 1 }, { 1, 0 }, { 0, 0 },
        { 0, 1 }, { 1, 1 }, { 1, 0 }, { 0, 0 }, { 0, 1 }, { 1, 1 },
        { 1, 0 }, { 0, 0 }, { 0, 1 }, { 1, 1 }, { 1, 0 }, { 0, 0 }
    };
    const std::vector<ScalarIndex3> triangles = {
        { 0, 1, 2 },    { 3, 0, 2 },    { 4, 5, 6 },    { 7, 4, 6 },
        { 8, 9, 10 },   { 11, 8, 10 },  { 12, 13, 14 }, { 15, 12, 14 },
        { 16, 17, 18 }, { 19, 16, 18 }, { 20, 21, 22 }, { 23, 20, 22 }
    };

public:
    Cube(const Properties &props) : Base(props) {
        m_face_count   = 12;
        m_vertex_count = 24;
        m_name         = "cube";

        m_faces_buf =
            DynamicBuffer<UInt32>::copy(triangles.data(), m_face_count * 3);
        m_vertex_positions_buf = empty<FloatStorage>(m_vertex_count * 3);
        m_vertex_normals_buf   = empty<FloatStorage>(m_vertex_count * 3);
        m_vertex_texcoords_buf = empty<FloatStorage>(m_vertex_count * 2);

        // TODO this is needed for the bbox(..) methods, but is it slower?
        m_faces_buf.managed();
        m_vertex_positions_buf.managed();
        m_vertex_normals_buf.managed();
        m_vertex_texcoords_buf.managed();

        for (uint8_t i = 0; i < m_vertex_count; ++i) {
            InputFloat *position_ptr = m_vertex_positions_buf.data() + i * 3;
            InputFloat *normal_ptr   = m_vertex_normals_buf.data() + i * 3;
            InputFloat *texcoord_ptr = m_vertex_texcoords_buf.data() + i * 2;

            InputPoint3f p  = vertices[i];
            InputNormal3f n = normals[i];
            p               = m_to_world.transform_affine(p);
            n               = normalize(m_to_world.transform_affine(n));
            m_bbox.expand(p);

            store_unaligned(position_ptr, p);
            store_unaligned(normal_ptr, n);
            store_unaligned(texcoord_ptr, texcoords[i]);
        }

        set_children();
    }

    MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS_VARIANT(Cube, Mesh)
MTS_EXPORT_PLUGIN(Cube, "Cube intersection primitive");
NAMESPACE_END(mitsuba)