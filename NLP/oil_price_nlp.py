import json
import os
from pathlib import Path

import jieba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.models import Word2Vec
from matplotlib.font_manager import FontProperties
from sklearn.decomposition import PCA


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def set_chinese_font() -> FontProperties | None:
    """尝试设置中文字体支持。"""
    try:
        windows_fonts = [
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simkai.ttf",
            "C:/Windows/Fonts/simsun.ttc",
        ]
        for font in windows_fonts:
            if os.path.exists(font):
                chinese_font = FontProperties(fname=font)
                plt.rcParams["font.family"] = chinese_font.get_name()
                plt.rcParams["axes.unicode_minus"] = False
                print(f"已设置中文字体: {chinese_font.get_name()}")
                return chinese_font
        print("警告: 未找到系统中文字体，将使用英文标签")
        return None
    except Exception as exc:
        print(f"字体设置错误: {exc}")
        return None


def preprocess_text(text: str) -> list[str]:
    """中文文本预处理和分词。"""
    text = str(text).replace("\n", " ").replace("\r", " ").replace("\t", " ")
    words = jieba.lcut(text)
    return [word for word in words if len(word) > 1]


def text_to_vector(tokens: list[str], model: Word2Vec) -> np.ndarray:
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(model.vector_size)


def visualize_word_vectors(
    words: list[str], model: Word2Vec, output_path: Path, font_prop: FontProperties | None = None
) -> None:
    vectors = []
    labels = []
    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])
            labels.append(word)

    if not vectors:
        print("没有找到关键词的向量表示")
        return

    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)

    plt.figure(figsize=(12, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])

    for idx, label in enumerate(labels):
        if font_prop:
            plt.annotate(label, (vectors_2d[idx, 0], vectors_2d[idx, 1]), fontproperties=font_prop)
        else:
            plt.annotate(label, (vectors_2d[idx, 0], vectors_2d[idx, 1]))

    plt.title("关键词向量可视化", fontproperties=font_prop)
    plt.xlabel("PCA 1", fontproperties=font_prop)
    plt.ylabel("PCA 2", fontproperties=font_prop)
    ensure_parent(output_path)
    plt.savefig(output_path)
    plt.close()


def visualize_word_vectors_english(words: list[str], model: Word2Vec, output_path: Path) -> None:
    vectors = []
    labels = []
    english_labels = {
        "减产": "Production_Cut",
        "制裁": "Sanction",
        "库存": "Inventory",
        "战争": "War",
        "协议": "Agreement",
        "价格": "Price",
        "需求": "Demand",
        "供应": "Supply",
        "OPEC": "OPEC",
        "俄罗斯": "Russia",
        "美国": "USA",
        "伊朗": "Iran",
    }

    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])
            labels.append(english_labels.get(word, word))

    if not vectors:
        print("没有找到关键词的向量表示")
        return

    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)

    plt.figure(figsize=(12, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])

    for idx, label in enumerate(labels):
        plt.annotate(label, (vectors_2d[idx, 0], vectors_2d[idx, 1]))

    plt.title("Keyword Vector Visualization")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    ensure_parent(output_path)
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    project_root = Path(__file__).resolve().parent
    config_path = project_root / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {config_path}")

    cfg = load_config(config_path)
    data_path = (project_root / cfg["paths"]["data_file"]).resolve()
    output_dir = (project_root / cfg["paths"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    chinese_font = set_chinese_font()

    df = pd.read_excel(data_path)
    text_column = cfg["columns"]["text_column"]
    change_column = cfg["columns"]["change_column"]

    if text_column not in df.columns:
        raise KeyError(f"缺少文本列: {text_column}")

    df["Tokenized"] = df[text_column].apply(preprocess_text)
    sentences = df["Tokenized"].tolist()

    model_cfg = cfg["word2vec"]
    model = Word2Vec(
        sentences=sentences,
        vector_size=model_cfg["vector_size"],
        window=model_cfg["window"],
        min_count=model_cfg["min_count"],
        workers=model_cfg["workers"],
        epochs=model_cfg["epochs"],
    )

    model_path = output_dir / cfg["outputs"]["model_file"]
    model.save(str(model_path))
    print(f"Word2Vec模型已保存至: {model_path}")

    df["Text_Vector"] = df["Tokenized"].apply(lambda tokens: text_to_vector(tokens, model))

    keywords = cfg["visualization"]["keywords"]
    visualize_word_vectors(
        keywords,
        model,
        output_dir / cfg["outputs"]["word_vectors_plot"],
        font_prop=chinese_font,
    )

    vector_df = pd.DataFrame(
        df["Text_Vector"].tolist(),
        columns=[f"Vec_{i}" for i in range(model.vector_size)],
    )
    final_df = pd.concat([df.drop(columns=["Tokenized", "Text_Vector"]), vector_df], axis=1)

    excel_output = output_dir / cfg["outputs"]["excel_with_vectors"]
    final_df.to_excel(excel_output, index=False)
    print(f"数据已保存至: {excel_output}")

    csv_output = output_dir / cfg["outputs"]["vectors_csv"]
    vector_df.to_csv(csv_output, index=False)
    print(f"文本向量已保存为CSV文件: {csv_output}")

    if change_column in final_df.columns:
        correlations = []
        for i in range(model.vector_size):
            corr = np.corrcoef(final_df[f"Vec_{i}"], final_df[change_column])[0, 1]
            correlations.append(corr)

        plt.figure(figsize=(15, 6))
        sns.heatmap([correlations], annot=False, cmap="coolwarm", cbar_kws={"label": "与价格变动的相关性"})
        if chinese_font:
            plt.title("词向量维度与油价变动的相关性", fontproperties=chinese_font)
        else:
            plt.title("Correlation between Vector Dimensions and Price Change")
        plt.xlabel("向量维度")
        plt.ylabel("")
        corr_path = output_dir / cfg["outputs"]["correlation_plot"]
        plt.savefig(corr_path)
        plt.close()
        print(f"相关性热力图已保存: {corr_path}")
    else:
        print(f"警告: 数据中缺少'{change_column}'列，无法计算相关性")

    if not chinese_font:
        print("尝试使用英文标签重新保存图像...")
        visualize_word_vectors_english(
            keywords,
            model,
            output_dir / cfg["outputs"]["word_vectors_english_plot"],
        )
        print("英文标签图像已保存")


if __name__ == "__main__":
    main()
