import streamlit as st

st.title("Live SCADA (placeholder)")
st.caption("This page is intentionally out of scope for now. I will keep the offline layer solid first.")

st.info(
    "When codes are ready, I will add:\n"
    "- a simulated real-time loop (stateful runner)\n"
    "- streaming plots\n"
    "- Modbus/TCP or file-based IO adapters\n"
    "- a minimal SCADA tag map\n"
)

try:
    from scada_playground_ui import render_scada_playground
    st.divider()
    st.subheader("Existing SCADA UI")
    render_scada_playground()
except Exception as e:
    st.warning("No SCADA module found (or import failed). This is expected for the offline-only milestone.")
    st.code(str(e))
