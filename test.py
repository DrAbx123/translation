import os
import re
import json
from typing import List, Dict, Any
import soundfile as sf
import pandas as pd

import torch
import whisperx
from pyannote.audio import Pipeline
from huggingface_hub import InferenceClient


HF_TOKEN = os.environ.get("HF_TOKEN", "")  # 从环境变量读取


def call_llm(messages: List[Dict[str, str]],
             model: str = "Qwen/Qwen3-32B",
             base_url: str | None = None,
             max_tokens: int = 800,
             temperature: float = 0.2) -> str:
    """
    用 HF_TOKEN 调用 Hugging Face 的 chat completion。
    - 如果你用 HF Inference / Inference Providers：base_url=None，model=模型名
    - 如果你有自己的 TGI/vLLM Endpoint：传 base_url="https://xxx"（OpenAI兼容 /v1/chat/completions）
    """
    # InferenceClient 的 chat.completions.create 是 OpenAI 风格 API :contentReference[oaicite:15]{index=15}
    client = InferenceClient(model=base_url or model, api_key=HF_TOKEN)
    resp = client.chat.completions.create(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content


def build_speaker_persona(samples_ja: str) -> Dict[str, Any]:
    prompt = [
        {"role": "system", "content": "你是资深日中本地化译者与角色对白编剧。输出必须是严格 JSON。"},
        {"role": "user", "content": f"""下面是一位说话人的日语台词片段（可能口语、断句不齐）：
{samples_ja}

请根据措辞、敬语、语气词、攻击性/温柔度、用词偏好，推断一个“可用于翻译时保持一致”的人设风格。
要求：
- 不要编造具体身份背景（除非台词强烈暗示），用“风格特征”描述即可
- 输出严格 JSON，字段：
  tone(简短), politeness(简短), preferred_pronouns(如有), catchphrases(数组), translation_rules(数组)
"""}
    ]
    out = call_llm(prompt, model="Qwen/Qwen3-30B-A3B-Instruct-2507", max_tokens=500, temperature=0.2)
    return json.loads(out)


def translate_with_persona(ja_text: str, persona: Dict[str, Any]) -> str:
    rules = "\n".join([f"- {r}" for r in persona.get("translation_rules", [])])
    prompt = [
        {"role": "system", "content": "你是日译中译者，目标是自然、准确、符合角色口吻。不要输出解释。"},
        {"role": "user", "content": f"""请把下面日语翻译成中文，并严格遵守角色口吻约束。

【角色口吻】
- tone: {persona.get("tone")}
- politeness: {persona.get("politeness")}
- preferred_pronouns: {persona.get("preferred_pronouns")}
- catchphrases: {persona.get("catchphrases")}
【额外翻译规则】
{rules if rules.strip() else "- 保持原意，不增删信息；口语化但不过度网络化。"}

【日语原文】
{ja_text}

【输出要求】
- 只输出中文译文，不要输出任何标签/解释
"""}
    ]
    return call_llm(prompt, model="Qwen/Qwen3-32B", max_tokens=800, temperature=0.2).strip()


def merge_words_to_utterances(segments: List[Dict[str, Any]], max_gap: float = 0.6) -> List[Dict[str, Any]]:
    """
    把 whisperx.assign_word_speakers 之后的 words 合成更自然的“话轮”。
    """
    end_punct = re.compile(r"[。！？!?]$")
    out = []

    cur = None

    def flush():
        nonlocal cur
        if not cur:
            return
        cur["ja"] = "".join(cur["ja_parts"]).strip()
        del cur["ja_parts"]
        out.append(cur)
        cur = None

    for seg in segments:
        words = seg.get("words") or []
        for w in words:
            spk = w.get("speaker", "UNKNOWN")
            ws, we = w.get("start"), w.get("end")
            token = w.get("word", "")

            if ws is None or we is None:
                continue

            if cur is None:
                cur = {"speaker": spk, "start": ws, "end": we, "ja_parts": [token]}
                continue

            gap = ws - cur["end"]
            speaker_changed = (spk != cur["speaker"])
            should_break = speaker_changed or (gap > max_gap) or end_punct.search(cur["ja_parts"][-1] if cur["ja_parts"] else "")

            if should_break:
                flush()
                cur = {"speaker": spk, "start": ws, "end": we, "ja_parts": [token]}
            else:
                cur["end"] = we
                cur["ja_parts"].append(token)

    flush()
    return out


def main(audio_wav_path: str, out_json_path: str = "result.json"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    # 1) WhisperX 转写（建议 large-v3，准确优先）:contentReference[oaicite:16]{index=16}
    model = whisperx.load_model("large-v3", device, compute_type=compute_type, vad_method="silero" )
    #model = whisperx.load_model("large-v3", device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_wav_path)
    result = model.transcribe(audio, batch_size=4, language="ja")

    # 2) 对齐（ja 默认会用 jonatasgrosman/wav2vec2-large-xlsr-53-japanese）:contentReference[oaicite:17]{index=17}
    model_a, metadata = whisperx.load_align_model(language_code="ja", device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # 3) pyannote diarization（community-1 + exclusive diarization）:contentReference[oaicite:18]{index=18}
    diar = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=HF_TOKEN)
    if device == "cuda":
        diar.to(torch.device("cuda"))

    # 读音频到内存（绕开 pyannote 内置解码器）
    wav, sr = sf.read(audio_wav_path, dtype="float32", always_2d=True)  # (T, C)
    wav = wav.T  # -> (C, T)
    waveform = torch.from_numpy(wav)
    if device == "cuda":
        waveform = waveform.to("cuda", non_blocking=True)
    diar_out = diar(
    {"waveform": waveform, "sample_rate": int(sr)},
    min_speakers=1,
    max_speakers=4,
    )

    # community-1 提供 exclusive diarization，更利于和转写对齐融合 :contentReference[oaicite:2]{index=2}
    diar_anno = diar_out.exclusive_speaker_diarization

    rows = []
    for seg, _, spk in diar_anno.itertracks(yield_label=True):
        rows.append({"start": float(seg.start), "end": float(seg.end), "speaker": str(spk)})
    diar_df = pd.DataFrame(rows)

    # 4) 词分配到 speaker
    result = whisperx.assign_word_speakers(diar_df, result)

    # 5) 合成“话轮”
    utterances = merge_words_to_utterances(result["segments"], max_gap=0.6)

    # 6) 为每个 speaker 建 persona（可选但你提了“人设”，建议做）
    by_spk = {}
    for u in utterances:
        by_spk.setdefault(u["speaker"], []).append(u["ja"])
    personas = {}
    for spk, texts in by_spk.items():
        sample = "\n".join(texts[:20])  # 取前 20 句做风格归纳
        personas[spk] = build_speaker_persona(sample)

    # 7) 翻译（按 speaker persona）
    for u in utterances:
        u["zh"] = translate_with_persona(u["ja"], personas.get(u["speaker"], {}))

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(utterances, f, ensure_ascii=False, indent=2)

    print(f"OK -> {out_json_path}")


if __name__ == "__main__":
    main("input_16k_mono.wav", "result.json")
