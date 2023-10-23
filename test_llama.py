import gradio as gr
from ctransformers import AutoModelForCausalLM


def load_llm():
    llm = AutoModelForCausalLM.from_pretrained(
        'llama-2-7b-chat.Q2_K.gguf',
        model_type  = 'llama',
        max_new_tokens = 1096,
        repetition_penalty = 1.13,
        temperature = 0.1
    )
    return llm
def llm_function(message, chat_history):
    llm = load_llm()
    print(message)
    response = llm(message)
    output_texts = response
    return output_texts

title = 'llama 7b demo'
examples = ['Will AI take over the world?',
            'Write python code for bubble sort',
            'Write a funny joke about software developers']

gr.ChatInterface(
    fn=llm_function,
    title=title,
    examples=examples
).launch()

