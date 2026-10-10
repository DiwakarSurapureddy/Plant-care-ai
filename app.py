"""Gradio Web Application for Plant Care AI."""
import gradio as gr
from src.chatbot import PlantCareChatbot
from src.config import DEFAULT_RETRIEVAL_MODE, RETRIEVAL_TOP_K

# Initialize core chatbot
chatbot_engine = PlantCareChatbot()


def format_ollama_display(status):
    """Format Ollama connection details and diagnostics."""
    if status.is_running and status.is_configured_model_available:
        return (
            f"**Status:** Online & Ready\n\n"
            f"- **Configured Model:** `{status.configured_model}` (Found locally)\n"
            f"- **Installed Models:** {', '.join(status.installed_models) if status.installed_models else 'None'}\n"
            f"- **Message:** {status.status_message}"
        )
    elif status.is_running:
        cmds = "\n".join([f"  - `{c}`" for c in status.recommended_commands])
        return (
            f"**Status:** Service Online, Model Missing\n\n"
            f"- **Configured Model:** `{status.configured_model}` (Not downloaded)\n"
            f"- **Installed Models:** {', '.join(status.installed_models) if status.installed_models else 'None'}\n"
            f"- **Instructions to install model:**\n{cmds}\n\n"
            f"*(Answers will use direct grounded extracts until the model is pulled)*"
        )
    else:
        cmds = "\n".join([f"  - `{c}`" for c in status.recommended_commands])
        return (
            f"**Status:** Offline (Connection Refused)\n\n"
            f"- **Message:** {status.status_message}\n\n"
            f"- **To start Ollama locally:**\n{cmds}\n\n"
            f"*(Answers will use direct grounded extracts from the knowledge base)*"
        )


def format_kb_stats(stats):
    """Format Knowledge Base statistics."""
    plants_str = ", ".join(stats["plants"]) if stats["plants"] else "None"
    err_str = f"\n- **Errors:** {', '.join(stats['load_error_details'])}" if stats["load_errors"] > 0 else ""
    return (
        f"- **Documents Indexed:** {stats['total_files']}\n"
        f"- **Knowledge Chunks:** {stats['total_chunks']}\n"
        f"- **Indexed Plants/Topics:** {plants_str}{err_str}"
    )


def handle_user_message(message, history, retrieval_mode, top_k):
    """Handle chat interaction and update sources/warnings panels."""
    if history is None:
        history = []

    user_text = (message or "").strip()
    if not user_text:
        return history, "", "", "", "", ""

    # Add user message to history
    new_history = list(history)
    new_history.append({"role": "user", "content": user_text})

    # Call chatbot
    resp = chatbot_engine.chat(
        user_query=user_text,
        retrieval_mode=retrieval_mode,
        top_k=int(top_k)
    )

    # Add assistant response to history
    new_history.append({"role": "assistant", "content": resp.answer})

    # Format sources panel
    if resp.source_documents:
        sources_text = "\n".join([f"- {s}" for s in resp.source_documents])
    else:
        sources_text = "No evidence retrieved."

    # Format trust warnings panel
    if resp.trust_warnings:
        warn_lines = []
        for w in resp.trust_warnings:
            icon = "[WARNING]" if w.severity == "warning" else ("[CAUTION]" if w.severity == "caution" else "[INFO]")
            warn_lines.append(f"- **{icon} `{w.source_file}`** ({w.chunk_heading}): {w.message}")
        warnings_text = "\n".join(warn_lines)
    else:
        warnings_text = "All retrieved evidence passed standard verification checks."

    # Format related plants
    if resp.related_plants:
        related_text = "\n".join([f"- {p}" for p in resp.related_plants])
    else:
        related_text = "No related plants documented for this query."

    # Diagnostics
    diag_lines = [
        f"- **Mode Requested:** `{retrieval_mode}`",
        f"- **Mode Used:** `{resp.retrieval_mode_used}`",
        f"- **Chunks Retrieved:** {resp.raw_evidence_count}"
    ]
    if resp.retrieval_fallback:
        diag_lines.append(f"- **Fallback Notice:** {resp.retrieval_diagnostic}")
    diag_text = "\n".join(diag_lines)

    # Re-fetch ollama status
    ollama_info = format_ollama_display(resp.ollama_status)

    return new_history, "", sources_text, warnings_text, related_text, diag_text


def rebuild_knowledge():
    """Rebuild knowledge index and return updated stats."""
    stats = chatbot_engine.rebuild_index()
    return format_kb_stats(stats)


def refresh_ollama():
    """Refresh Ollama status display."""
    status = chatbot_engine.get_ollama_status()
    return format_ollama_display(status)


# Build Gradio UI
with gr.Blocks(title="Plant Care AI") as demo:
    gr.Markdown(
        """
        # 🌱 Plant Care AI
        ### Local RAG-powered Nursery & Plant Care Knowledge Assistant
        *Grounding plant care guidance in structured OKF Markdown files with local embeddings & Ollama.*
        """
    )

    with gr.Row():
        # Left Column: Chat and input
        with gr.Column(scale=7):
            chatbot_ui = gr.Chatbot(
                value=[],
                label="Conversation",
                height=460
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask about plant care, watering, sunlight, soil, problems, or nursery prices...",
                    label="Your Plant Question",
                    lines=1,
                    scale=9
                )
                send_btn = gr.Button("Submit", variant="primary", scale=2)
                clear_btn = gr.Button("Clear", scale=1)

            gr.Markdown("**Example Questions:**")
            with gr.Row():
                ex1 = gr.Button("How do I care for a Money Plant?", size="sm")
                ex2 = gr.Button("How much sunlight does a Rose need?", size="sm")
                ex3 = gr.Button("Why are my Hibiscus leaves turning yellow?", size="sm")
            with gr.Row():
                ex4 = gr.Button("How can I grow Tulsi at home?", size="sm")
                ex5 = gr.Button("What is the price of an Aloe Vera plant?", size="sm")
                ex6 = gr.Button("Suggest plants similar to Money Plant.", size="sm")
                ex7 = gr.Button("Compare Rose and Hibiscus care requirements.", size="sm")

        # Right Column: Controls, Sources, Trust & Status
        with gr.Column(scale=5):
            with gr.Accordion("⚙️ Search & Retrieval Controls", open=True):
                mode_selector = gr.Dropdown(
                    choices=["hybrid", "semantic", "keyword"],
                    value=DEFAULT_RETRIEVAL_MODE,
                    label="Retrieval Mode",
                    info="Choose retrieval strategy (Keyword, Semantic vector search, or Hybrid)"
                )
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=8,
                    value=RETRIEVAL_TOP_K,
                    step=1,
                    label="Number of Retrieved Chunks (Top-K)"
                )

            with gr.Accordion("📚 Retrieved Sources & Attribution", open=True):
                sources_box = gr.Markdown("No queries executed yet.")

            with gr.Accordion("🛡️ Trust & Freshness Warnings", open=True):
                warnings_box = gr.Markdown("Status and verification checks will appear here.")

            with gr.Accordion("🌿 Related Plants", open=False):
                related_box = gr.Markdown("Related plant suggestions will appear here.")

            with gr.Accordion("🔍 Retrieval Diagnostics", open=False):
                diag_box = gr.Markdown("Diagnostics will appear here.")

            with gr.Accordion("🦙 Ollama Local LLM Status", open=False):
                initial_status = chatbot_engine.get_ollama_status()
                ollama_status_box = gr.Markdown(format_ollama_display(initial_status))
                refresh_ollama_btn = gr.Button("Refresh Ollama Status", size="sm")

            with gr.Accordion("📊 Knowledge Base Statistics", open=False):
                initial_stats = chatbot_engine.get_stats()
                kb_stats_box = gr.Markdown(format_kb_stats(initial_stats))
                rebuild_btn = gr.Button("Rebuild Knowledge Index", size="sm")

    # Wire event handlers
    submit_outputs = [chatbot_ui, msg_input, sources_box, warnings_box, related_box, diag_box]

    send_btn.click(
        fn=handle_user_message,
        inputs=[msg_input, chatbot_ui, mode_selector, top_k_slider],
        outputs=submit_outputs
    )

    msg_input.submit(
        fn=handle_user_message,
        inputs=[msg_input, chatbot_ui, mode_selector, top_k_slider],
        outputs=submit_outputs
    )

    clear_btn.click(lambda: ([], "", "No evidence retrieved.", "Cleared.", "", ""), None, submit_outputs)

    # Example button wiring
    def set_and_submit(example_text, history, mode, k):
        return handle_user_message(example_text, history, mode, k)

    for btn in [ex1, ex2, ex3, ex4, ex5, ex6, ex7]:
        btn.click(
            fn=set_and_submit,
            inputs=[btn, chatbot_ui, mode_selector, top_k_slider],
            outputs=submit_outputs
        )

    rebuild_btn.click(fn=rebuild_knowledge, inputs=[], outputs=[kb_stats_box])
    refresh_ollama_btn.click(fn=refresh_ollama, inputs=[], outputs=[ollama_status_box])


if __name__ == "__main__":
    import socket
    import os

    def find_open_port(default_port: int = 7860) -> int:
        env_p = os.getenv("GRADIO_SERVER_PORT")
        if env_p and env_p.isdigit():
            return int(env_p)
        for p in range(default_port, default_port + 20):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", p)) != 0:
                    return p
        return default_port

    active_port = find_open_port()
    print(f"Starting Plant Care AI on http://127.0.0.1:{active_port}")
    demo.launch(server_name="127.0.0.1", server_port=active_port, share=False)

