from flask import Flask, jsonify

from server.ai_main import main_ai
app = Flask(__name__)


@app.route('/run', methods=['GET'])
def run_main():
    result = main_ai(target_date='2021-01-30')  # main 함수 실행
    return jsonify(result.to_dict(orient='records'))  # JSON 형태로 변환하여 반환

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
