import serial  # 시리얼 통신을 위한 모듈 임포트
import time

class SerialCommunication:
    def __init__(self, port='/dev/ttyS0', baudrate=4800, timeout=0.0001):
        """시리얼 포트를 초기화하는 생성자"""
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        self.ser.flush()  # 시리얼 포트의 수신 버퍼를 비움

    def receive_data(self):
        """시리얼 포트에서 수신된 모든 데이터를 읽어오는 메서드"""
        time.sleep(0.2)  # 데이터를 모두 수신할 시간을 줌
        data_in_waiting = self.ser.inWaiting()  # 수신된 데이터의 크기를 확인
        if data_in_waiting > 0:  # 수신된 데이터가 있는지 확인
            result = self.ser.read(data_in_waiting)  # 수신된 모든 데이터를 읽어옴
            rx_values = list(result)  # 바이트 데이터를 정수 리스트로 변환
            print(f"RX: {rx_values}")  # 수신된 데이터를 출력
            return rx_values  # 수신된 데이터를 리스트로 반환
        else:
            return []  # 데이터가 없으면 빈 리스트를 반환

    def send_data(self, data):
        """10진수 숫자를 2진수로 변환하여 8비트씩 쪼개어 송신하는 메서드"""
        high_byte = (data >> 8) & 0xFF  # 상위 8비트 추출
        low_byte = data & 0xFF  # 하위 8비트 추출
        self.ser.write(serial.to_bytes([high_byte]))  # 상위 바이트 송신
        self.ser.write(serial.to_bytes([low_byte]))  # 하위 바이트 송신
        print(f"TX: {high_byte} (2진수: {high_byte:08b}), {low_byte} (2진수: {low_byte:08b})")  # 송신한 데이터를 출력

    def close(self):
        """시리얼 포트를 닫는 메서드"""
        self.ser.close()
        print("시리얼 포트를 닫습니다.")


# 메인 루프: 사용자 입력을 받아서 송신하고, 'q'를 입력하면 종료
def main():
    serial_com = SerialCommunication()  # 클래스 인스턴스를 생성

    while True:
        user_input = input("송신할 숫자를 입력하세요 (종료하려면 'q'): ")

        if user_input.lower() == 'q':  # 'q' 또는 'Q'를 입력하면 종료
            print("프로그램을 종료합니다.")
            serial_com.close()
            break

        try:
            data = int(user_input)  # 입력된 값을 정수로 변환
            if 0 <= data <= 65535:  # 데이터가 0~65535 범위 내에 있는지 확인
                serial_com.send_data(data)  # 데이터를 송신
                time.sleep(0.3)  # 약간의 대기 시간을 줌
                serial_com.receive_data()  # 수신된 데이터 확인
            else:
                print("0에서 65535 사이의 숫자를 입력하세요.")
        except ValueError:
            print("유효한 숫자를 입력하세요.")

if __name__ == "__main__":
    main()
