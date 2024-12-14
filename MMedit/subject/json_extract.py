import json
import os

def filter_json(input_file, output_file, keys_to_keep):
    """
    读取输入的JSON文件，过滤每个字典，仅保留指定的键，并将结果写入输出文件。

    :param input_file: 输入的JSON文件路径
    :param output_file: 输出的JSON文件路径
    :param keys_to_keep: 需要保留的键列表
    """
    try:
        # 检查输入文件是否存在
        if not os.path.isfile(input_file):
            print(f"错误：输入文件 {input_file} 不存在。")
            return

        # 读取整个JSON数据
        with open(input_file, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
        
        # 确保数据是一个列表
        if not isinstance(data, list):
            print(f"错误：期望在 {input_file} 中找到一个字典列表。")
            return
        
        # 过滤每个字典，只保留指定的键
        filtered_data = []
        for index, item in enumerate(data):
            if isinstance(item, dict):
                filtered_item = {key: item[key] for key in keys_to_keep if key in item}
                filtered_data.append(filtered_item)
            else:
                print(f"警告：跳过非字典项，第 {index} 个项: {item}")
        
        # 将过滤后的数据写入输出文件
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(filtered_data, outfile, ensure_ascii=False, indent=4)
        
        print(f"过滤后的数据已保存到 {output_file}")
    
    except json.JSONDecodeError as e:
        print(f"JSON解码错误：{e}")
    except Exception as e:
        print(f"发生错误：{e}")

if __name__ == "__main__":
    # 定义输入和输出文件路径
    input_file = "/data/lishichao/data/model_edit/editing-data/vqa/vqa_train.json"  # 替换为您的原始JSON文件路径
    output_file = "/data/lishichao/data/model_edit/editing-data/vqa/vqa_train_extract.json"  # 替换为您希望保存的新JSON文件路径

    # 定义需要保留的键
    keys_to_keep = ["src", "pred", "image"]

    # 调用函数进行过滤
    # filter_json(input_file, output_file, keys_to_keep)