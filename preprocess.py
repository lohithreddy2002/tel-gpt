from tqdm  import tqdm
from indicnlp.tokenize.sentence_tokenize import sentence_split 
import argparse
 
def read_file(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    print("read_file")
    return lines



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", help = "file for trainging tokenizer ",default="/home/agv/Documents/tel00.txt")
    parser.add_argument("--lang", help = "language of training data ",default="te")
    parser.add_argument("--output_path", help = "path to save preprocessed file",default="/home/agv/Documents/te_out.txt")

    args = parser.parse_args()
    lines = read_file(args.input_files)
    lang = args.lang
    split_lines = [sentence_split(line, lang) for line, lang in tqdm(zip(lines, [lang] * len(lines)))]
    print("splitting lines done")
    with open(args.output_file, 'w') as f:
            for paragraph in split_lines:
                for line in paragraph:
                    if len(line) > 5:
                        f.write(line.strip() + '\n')
                f.write('\n')
