import os
import json
import glob
from pathlib import Path
from faster_whisper import WhisperModel
from tqdm import tqdm
import torch

DATASET_ROOT = r"./public/question-bank"
OUTPUT_ROOT_DIR = r"./public/question-bank/speech_to_text_results_lecture"
LOCAL_MODEL_PATH = r"./models/whisper-base.en"
DEVICE = "auto"
COMPUTE_TYPE = "float16"
PROCESS_INTERVAL = 50

def verify_model_files():
    required_files = [
        "config.json", "model.bin", "preprocessor_config.json",
        "tokenizer.json", "tokenizer_config.json", "vocabulary.txt"
    ]
    missing_files = []

    print("=" * 50)
    print("📁 验证模型文件完整性...")
    for file_name in required_files:
        file_path = os.path.join(LOCAL_MODEL_PATH, file_name)
        if os.path.exists(file_path):
            print(f"✅ 存在：{file_name}")
        else:
            print(f"❌ 缺失：{file_name}（路径：{file_path}）")
            missing_files.append(file_name)

    if missing_files:
        print(f"\n❌ 错误：缺失关键文件！{missing_files}")
        print("=" * 50)
        exit(1)
    print(f"\n✅ 所有{len(required_files)}个模型文件验证通过！")
    print("=" * 50)

def verify_gpu_environment():
    print("\n" + "=" * 50)
    print("🖥️  检查GPU/CUDA环境...")

    global COMPUTE_TYPE

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

        print(f"✅ GPU设备：{gpu_name}")
        print(f"✅ CUDA版本：{cuda_version}（匹配12.2）")
        print(f"✅ GPU显存：{gpu_memory:.1f}GB")
        print(f"✅ 计算类型：{COMPUTE_TYPE}（RTX 4060优化）")
        print("✅ GPU环境验证通过！")
    else:
        print("⚠️  未检测到GPU，切换到CPU模式！")
        COMPUTE_TYPE = "int8"
    print("=" * 50)

def clean_task_text(raw_task_text):
    if not raw_task_text:
        return "无任务描述"

    clean_text = raw_task_text
    if "\n直接做题" in clean_text:
        clean_text = clean_text.split("\n直接做题")[0]

    else:
        redundant_keywords = [
            "直接做题", "新建笔记", "我的笔记", "编辑笔记", "精华内容",
            "优秀录音", "网友思路", "名师思路", "分数最高", "最新",
            "会员福利内容准备中", "题目讨论", "已经输入", "标记为提问",
            "答案或思路", "发表", "相关题型其他题目",
            "查看听力原文", "听力原文", "查看原文", "Transcript",
            "查看解析", "名师解析", "满分答案", "音频播放", "下载音频",
            "收藏题目", "报错题目", "加入错题本", "分享题目"
        ]
        for keyword in redundant_keywords:
            if keyword in clean_text:
                clean_text = clean_text.split(keyword)[0]
                break

    clean_text = clean_text.strip().replace("\n", "").replace("\r", "").replace("  ", " ")
    return clean_text if clean_text else "无有效任务描述"


def get_base_lecture_task_info(official_id):
    lecture_json_path = Path(DATASET_ROOT) / "lecture" / f"{official_id}.json"
    hearing_text = ""
    if lecture_json_path.exists():
        try:
            with open(lecture_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            hearing_text = data.get("lecture_text", "").strip()
        except Exception as e:
            print(f"⚠️  读取lecture JSON失败 {lecture_json_path}：{str(e)[:50]}...")

    task_json_path = Path(DATASET_ROOT) / "task" / f"{official_id}.json"
    raw_task = ""
    if task_json_path.exists():
        try:
            with open(task_json_path, 'r', encoding='utf-8') as tf:
                task_data = json.load(tf)
            raw_task = task_data.get("task", "")
        except Exception as e:
            print(f"⚠️  读取task JSON失败 {task_json_path}：{str(e)[:50]}...")
    clean_task = clean_task_text(raw_task)

    return hearing_text, clean_task

def read_answer_json(answer_json_path, audio_file_name):
    score = "无评分"
    try:
        with open(answer_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        records = data.get("records", [])
        for record in records:
            local_mp3 = record.get("local_mp3", "")
            if audio_file_name in local_mp3:
                score_val = record.get("score")
                score = f"{score_val}分" if score_val is not None else "无评分"
                break
    except Exception as e:
        error_msg = str(e)[:50] + "..." if len(str(e)) > 50 else str(e)
        print(f"⚠️  读取answer JSON失败 {os.path.basename(answer_json_path)}：{error_msg}")
    return score

def transcribe_single_audio(model, audio_path):
    try:
        segments, _ = model.transcribe(
            audio_path,
            language="en",
            beam_size=4,
            vad_filter=True,
            vad_parameters={"threshold": 0.6},
            without_timestamps=True
        )
        transcribed_text = " ".join([seg.text.strip() for seg in segments]).strip()
        return transcribed_text
    except Exception as e:
        error_msg = str(e)[:50] + "..." if len(str(e)) > 50 else str(e)
        print(f"⚠️  处理失败 {os.path.basename(audio_path)}：{error_msg}")
        return ""

def write_lecture_files(official_id, hearing_text, clean_task):
    official_folder = os.path.join(OUTPUT_ROOT_DIR, official_id)
    os.makedirs(official_folder, exist_ok=True)

    hearing_file = os.path.join(official_folder, "听力原文.txt")
    with open(hearing_file, 'w', encoding='utf-8') as f:
        f.write(hearing_text if hearing_text else "无听力原文")

    task_file = os.path.join(official_folder, "任务描述.txt")
    with open(task_file, 'w', encoding='utf-8') as f:
        f.write(clean_task)

    return True


def write_answer_files(official_id, audio_file_name, transcribed_text, score):
    official_folder = os.path.join(OUTPUT_ROOT_DIR, official_id)
    student_audio_folder = os.path.join(official_folder, "考生语音转文本")
    os.makedirs(student_audio_folder, exist_ok=True)

    trans_file = os.path.join(student_audio_folder, f"{audio_file_name}.txt")
    with open(trans_file, 'w', encoding='utf-8') as f:
        f.write(transcribed_text if transcribed_text else "转写失败")

    score_file = os.path.join(student_audio_folder, f"{audio_file_name}_评分.txt")
    with open(score_file, 'w', encoding='utf-8') as f:
        f.write(score)

    return True


def process_lecture_audio():
    print("\n" + "=" * 50)
    lecture_dir = os.path.join(DATASET_ROOT, "lecture")
    print(f"🔍 查找Lecture数据集：{lecture_dir}")

    lecture_json_files = glob.glob(os.path.join(lecture_dir, "*.json"))
    official_ids = [Path(f).stem for f in lecture_json_files]

    if not official_ids:
        print(f"⚠️  Lecture目录下未找到JSON文件！路径：{lecture_dir}")
    else:
        total_ids = len(official_ids)
        print(f"✅ 找到 {total_ids} 个Lecture官方ID")

        success_write_count = 0
        processed_count = 0
        for official_id in tqdm(official_ids, desc="Lecture基础信息写入进度", unit="id"):
            hearing_text, clean_task = get_base_lecture_task_info(official_id)

            if write_lecture_files(official_id, hearing_text, clean_task):
                success_write_count += 1

            processed_count += 1
            if processed_count % PROCESS_INTERVAL == 0:
                print(
                    f"\n📥 Lecture已处理 {processed_count}/{total_ids} 个ID，成功写入 {success_write_count} 个！")

        print(f"\n📊 Lecture处理统计：")
        print(f"   • 总ID数：{total_ids} 个")
        print(f"   • 成功写入文件夹：{success_write_count} 个")
        success_rate = (success_write_count / total_ids) * 100 if total_ids > 0 else 0
        print(f"   • 写入成功率：{success_rate:.1f}%")
    print("=" * 50)

def process_answer_audio(model):
    print("\n" + "=" * 50)
    answer_dir = os.path.join(DATASET_ROOT, "answer")
    print(f"🔍 查找Answer数据集：{answer_dir}")

    audio_files = []
    audio_files.extend(glob.glob(os.path.join(answer_dir, "*", "*.mp3"), recursive=True))
    audio_files.extend(glob.glob(os.path.join(answer_dir, "*", "*.wav"), recursive=True))

    if not audio_files:
        print(f"⚠️  Answer目录下未找到MP3/WAV文件！路径：{answer_dir}")
    else:
        total_files = len(audio_files)
        print(f"✅ 找到 {total_files} 个Answer音频文件（MP3/WAV）")

        success_write_count = 0
        processed_count = 0
        for audio_path in tqdm(audio_files, desc="Answer音频转换进度", unit="file"):
            audio_path_obj = Path(audio_path)
            official_id = audio_path_obj.parent.name
            audio_file_name = audio_path_obj.name
            answer_json_path = audio_path_obj.parent / "1.json"

            score = read_answer_json(str(answer_json_path), audio_file_name)

            transcribed_text = transcribe_single_audio(model, str(audio_path))

            if write_answer_files(official_id, audio_file_name, transcribed_text, score):
                success_write_count += 1

            processed_count += 1
            if processed_count % PROCESS_INTERVAL == 0:
                print(
                    f"\n📥 Answer已处理 {processed_count}/{total_files} 个文件，成功写入 {success_write_count} 个文件夹！")

        print(f"\n📊 Answer处理统计：")
        print(f"   • 总文件数：{total_files} 个")
        print(f"   • 成功写入文件夹：{success_write_count} 个")
        success_rate = (success_write_count / total_files) * 100 if total_files > 0 else 0
        print(f"   • 写入成功率：{success_rate:.1f}%")
    print("=" * 50)


def main():
    verify_model_files()
    verify_gpu_environment()

    print("\n" + "=" * 50)
    print("🚀 加载Whisper模型（GPU加速）...")
    try:
        model = WhisperModel(
            model_size_or_path=LOCAL_MODEL_PATH,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            download_root=None
        )
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败：{str(e)}")
        print("💡 解决方案：检查model.bin是否损坏，或重新下载")
        print("=" * 50)
        exit(1)
    print("=" * 50)

    process_lecture_audio()

    process_answer_audio(model)

    print("\n" + "=" * 50)
    print("🎉 所有处理完成！")
    print(f"\n💾 结果根路径：{OUTPUT_ROOT_DIR}")
    print("📁 最终文件夹结构（示例）：")
    print(f"   {OUTPUT_ROOT_DIR}/")
    print(f"   ├─ 001/                          # official_id=001的文件夹")
    print(f"   │  ├─ 听力原文.txt               # 官方听力原文")
    print(f"   │  ├─ 任务描述.txt               # 清理后的任务描述")
    print(f"   │  └─ 考生语音转文本/            # 该题下所有考生音频结果")
    print(f"   │     ├─ 1_01.mp3.txt            # 考生音频1_01.mp3的转写内容")
    print(f"   │     ├─ 1_01.mp3_评分.txt       # 对应音频的评分")
    print(f"   │     ├─ 1_02.wav.txt            # 考生音频1_02.wav的转写内容")
    print(f"   │     └─ 1_02.wav_评分.txt       # 对应音频的评分")
    print(f"   ├─ 002/                          # official_id=002的文件夹")
    print(f"   │  └─ ...")
    print(f"   └─ 054/                          # official_id=054的文件夹")
    print("=" * 50)


if __name__ == "__main__":
    main()