import argparse
import sys

from mz import play

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-l", "--list-devices", action="store_true",
        help="Show list of audio output devices and exit.")
    args, remaining = parser.parse_known_args()
    if args.list_devices:
        play.list_devices()
        sys.exit(0)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parser])
    parser.add_argument(
        "-f", "--modules_file_name", required=True,
        help="The file to load output_gen from, and to watch and react to saves!")
    parser.add_argument(
        "-c", "--output_gen_class", required=True,
        help="A `Module` subclass in MODULES_FILE_NAME.")
    parser.add_argument(
        "-d", "--device", type=int, default=-1,
        help="Output device (numeric ID or substring, -1 to auto-detect.)")
    parser.add_argument(
        "--midi_knobs", type=str, default="traktor_kontrol_knobs.txt",
        help=("File to read midi assignments from, see `traktor_kontrol_knobs.txt` for an example. "
              "Use `midi_lib.py ... map_device` to create a mapping."))
    parser.add_argument(
        "--midi_port_name_regex", type=str, default="Traktor Kontrol",
        help="Regex matching a connected midi device. To see devices, use `midi_lib.py list_devices`.")

    parser.add_argument("--record_max_minutes", type=int, default=5)
    args = parser.parse_args(remaining)

    play.start_sound_loop(
        modules_file_name=args.modules_file_name,
        output_gen_class=args.output_gen_class,
        device=args.device,
        record_max_minutes=args.record_max_minutes,
        midi_knobs_file=args.midi_knobs,
        midi_port_name_regex=args.midi_port_name_regex)


if __name__ == "__main__":
    main()

