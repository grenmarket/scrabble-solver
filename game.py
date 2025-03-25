from flask import Flask, request, jsonify

from board import BoardData

app = Flask(__name__)

@app.route('/board', methods=['POST'])
def receive_board():
    try:
        data = request.json

        board_data = BoardData.from_dict(data)

        print(f"Received board data with timestamp: {board_data.timestamp}")

        return jsonify({"status": "success", "message": "Board data received"}), 200

    except KeyError as e:
        return jsonify({"error": f"Missing required field: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)