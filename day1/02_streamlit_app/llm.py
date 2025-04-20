# llm.py
import os
import torch
from transformers import pipeline
import streamlit as st
import time
from config import MODEL_NAME
from huggingface_hub import login

# モデルをキャッシュして再利用
@st.cache_resource
def load_model():
    """LLMモデルをロードする"""
    try:

        # アクセストークンを保存
        hf_token = st.secrets["huggingface"]["token"]
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {device}") # 使用デバイスを表示
        pipe = pipeline(
            "text-generation",
            model=MODEL_NAME,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device
        )
        st.success(f"モデル '{MODEL_NAME}' の読み込みに成功しました。")
        return pipe
    except Exception as e:
        st.error(f"モデル '{MODEL_NAME}' の読み込みに失敗しました: {e}")
        st.error("GPUメモリ不足の可能性があります。不要なプロセスを終了するか、より小さいモデルの使用を検討してください。")
        return None

def generate_response(pipe, user_question, max_new_tokens, temperature, top_p):
    """LLMを使用して質問に対する回答を生成する"""
    if pipe is None:
        return "モデルがロードされていないため、回答を生成できません。", 0

    try:
        start_time = time.time()
        messages = [
            {"role": "user", "content": user_question},
        ]
        # 受け取ったパラメータで出力を生成
        outputs = pipe(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p
        )
        # 出力から回答を抽出する（既存の処理）
        assistant_response = ""
        if outputs and isinstance(outputs, list) and outputs[0].get("generated_text"):
            if isinstance(outputs[0]["generated_text"], list) and len(outputs[0]["generated_text"]) > 0:
                last_message = outputs[0]["generated_text"][-1]
                if last_message.get("role") == "assistant":
                    assistant_response = last_message.get("content", "").strip()
            elif isinstance(outputs[0]["generated_text"], str):
                full_text = outputs[0]["generated_text"]
                prompt_end = user_question
                response_start_index = full_text.find(prompt_end) + len(prompt_end)
                possible_response = full_text[response_start_index:].strip()
                if "<start_of_turn>model" in possible_response:
                    assistant_response = possible_response.split("<start_of_turn>model\n")[-1].strip()
                else:
                    assistant_response = possible_response

        if not assistant_response:
            print("Warning: Could not extract assistant response. Full output:", outputs)
            assistant_response = "回答の抽出に失敗しました。"

        end_time = time.time()
        response_time = end_time - start_time
        print(f"Generated response in {response_time:.2f}s")
        return assistant_response, response_time

    except Exception as e:
        st.error(f"回答生成中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return f"エラーが発生しました: {str(e)}", 0