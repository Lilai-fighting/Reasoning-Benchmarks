import json
from pathlib import Path

SYSTEM_PROMPT = (
    "Map each natural language command to its targeted output action sequence. "
    "Return only the target output."
)

def parse_scan_line(line: str):
    """
    Parse one SCAN line of the form:
    IN: walk after run OUT: I_RUN I_WALK
    """
    line = line.strip()
    if not line:
        return None

    if not line.startswith("IN: ") or " OUT: " not in line:
        raise ValueError(f"Invalid SCAN line: {line}")

    command_part, target_part = line.split(" OUT: ", maxsplit=1)
    command = command_part.replace("IN: ", "", 1).strip()
    target = target_part.strip()

    return command, target

def convert_scan_file(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f_in, \
         output_path.open("w", encoding="utf-8") as f_out:

        valid_index = 0

        for line in f_in:
            parsed = parse_scan_line(line)

            if parsed is None:
                continue

            input_command, output_sequence = parsed

            example = {
                "index": valid_index,
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": input_command
                    },
                    {
                        "role": "assistant",
                        "content": output_sequence
                    }
                ]
            }

            f_out.write(json.dumps(example, ensure_ascii=False) + "\n")
            valid_index += 1

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    convert_scan_file(args.input_path, args.output_path)

