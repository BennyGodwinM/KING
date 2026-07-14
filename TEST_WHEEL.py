import time
import serial

SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200
TEST_DURATION = 1.5  # seconds per test


def send_command(ser: serial.Serial, command: str) -> None:
    ser.write(command.encode("ascii"))
    ser.flush()
    print(f"Sent: {command}")


def run_test(
    ser: serial.Serial,
    command: str,
    description: str,
    duration: float = TEST_DURATION,
) -> None:
    input(f"\nPress ENTER to test {description}...")
    send_command(ser, command)
    time.sleep(duration)
    send_command(ser, "S")
    time.sleep(0.5)
    print(f"{description} test finished.")


def main() -> None:
    try:
        with serial.Serial(
            SERIAL_PORT,
            BAUD_RATE,
            timeout=0.1,
        ) as ser:
            time.sleep(2.0)

            print("\nROBOT WHEEL TEST")
            print("Lift the robot so all four wheels are off the ground.")
            print("Each command runs briefly and then automatically stops.")
            print("Press Ctrl+C at any time to stop.\n")

            while True:
                print("\nChoose a test:")
                print("1 = Forward")
                print("2 = Left turn")
                print("3 = Right turn")
                print("4 = Run all tests")
                print("s = Stop")
                print("q = Quit")

                choice = input("> ").strip().lower()

                if choice == "1":
                    run_test(ser, "F", "FORWARD")

                elif choice == "2":
                    run_test(ser, "L", "LEFT TURN")

                elif choice == "3":
                    run_test(ser, "R", "RIGHT TURN")

                elif choice == "4":
                    run_test(ser, "F", "FORWARD")
                    run_test(ser, "L", "LEFT TURN")
                    run_test(ser, "R", "RIGHT TURN")

                elif choice == "s":
                    send_command(ser, "S")

                elif choice == "q":
                    send_command(ser, "S")
                    print("Exiting.")
                    break

                else:
                    print("Invalid selection.")

    except KeyboardInterrupt:
        print("\nEmergency stop requested.")
        try:
            with serial.Serial(
                SERIAL_PORT,
                BAUD_RATE,
                timeout=0.1,
            ) as ser:
                send_command(ser, "S")
        except Exception:
            pass

    except serial.SerialException as exc:
        print(f"Serial error: {exc}")
        print(f"Check that the Arduino is connected at {SERIAL_PORT}.")


if __name__ == "__main__":
    main()