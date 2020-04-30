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
mesh plugin.

*/
template <typename Float, typename Spectrum>
class Cube final : public Mesh<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Mesh, m_face_count, m_faces, m_face_size, m_face_struct,
                    m_vertex_count, m_vertices, m_vertex_size, m_vertex_struct,
                    m_normal_offset, m_texcoord_offset, vertex, m_to_world,
                    recompute_vertex_normals, m_disable_vertex_normals, m_name,
                    m_bbox, emitter, is_emitter, sensor, is_sensor)
    MTS_IMPORT_TYPES()

    using typename Base::FaceHolder;
    using typename Base::InputFloat;
    using typename Base::InputNormal3f;
    using typename Base::InputPoint3f;
    using typename Base::InputVector2f;
    using typename Base::InputVector3f;
    using typename Base::ScalarIndex;
    using typename Base::ScalarSize;
    using typename Base::VertexHolder;
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
        
        m_vertex_struct = new Struct();
        for (auto name : { "x", "y", "z" })
            m_vertex_struct->append(name, struct_type_v<InputFloat>);
        for (auto name : { "nx", "ny", "nz" })
            m_vertex_struct->append(name, struct_type_v<InputFloat>);
        m_normal_offset = (ScalarIndex) m_vertex_struct->offset("nx");
        for (auto name : { "u", "v" })
            m_vertex_struct->append(name, struct_type_v<InputFloat>);
        m_texcoord_offset = (ScalarIndex) m_vertex_struct->offset("u");

        m_face_struct = new Struct();
        for (size_t i = 0; i < 3; ++i)
            m_face_struct->append(tfm::format("i%i", i),
                                  struct_type_v<ScalarIndex>);

        // x, y, z, nx, ny, nz, u, v are stored in the vertex buffer as float32
        m_vertex_size = 8 * 4;
        // faces are defined as 3 uint32 values
        m_face_size = 3 * 4;

        m_vertices = VertexHolder(new uint8_t[(m_vertex_count) *m_vertex_size]);
        m_faces    = FaceHolder(new uint8_t[(m_face_count) *m_face_size]);
        m_normal_offset   = 3 * 4;
        m_texcoord_offset = 6 * 4;

        memcpy(m_faces.get(), triangles.data(), m_face_count * m_face_size);

        for (uint8_t i = 0; i < m_vertex_count; ++i) {
            uint8_t *vertex_ptr = vertex(i);

            InputPoint3f p  = vertices[i];
            InputNormal3f n = normals[i];
            p               = m_to_world.transform_affine(p);
            n               = normalize(m_to_world.transform_affine(n));
            m_bbox.expand(p);

            store_unaligned(vertex_ptr, p);
            store_unaligned(vertex_ptr + m_normal_offset, n);
            store_unaligned(vertex_ptr + m_texcoord_offset, texcoords[i]);
        }

        if (!m_disable_vertex_normals && normals.empty())
            recompute_vertex_normals();

        if (is_emitter())
            emitter()->set_shape(this);
        if (is_sensor())
            sensor()->set_shape(this);
    }

    MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS_VARIANT(Cube, Mesh)
MTS_EXPORT_PLUGIN(Cube, "Cube intersection primitive");
NAMESPACE_END(mitsuba)