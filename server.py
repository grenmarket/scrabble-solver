import time

from flask import Flask, request, jsonify
from flask_cors import CORS
from board import ScrabbleBoard, Tile

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
board = ScrabbleBoard()
t1 = time.time_ns() // 1000000
board.load()
t2 = time.time_ns() // 1000000
print(f'board created, {(t2-t1)//1000}s')

@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/board', methods=['POST'])
def receive_board():
    try:
        data = request.json

        # Verify that the required fields are present
        if 'board' not in data:
            return jsonify({"error": "Missing 'board' field"}), 400

        if 'rack' not in data:
            return jsonify({"error": "Missing 'rack' field"}), 400

        # Process board data
        board_data = data['board']
        rack_data = data['rack']

        processed_board = []
        for row in board_data:
            processed_row = []
            for tile_data in row:
                if tile_data is not None:
                    processed_row.append(Tile(
                        letter=tile_data.get('letter', ''),
                        is_blank=tile_data.get('isBlank', False)
                    ))
                else:
                    processed_row.append(None)
            processed_board.append(processed_row)

        # Convert rack data to Tile objects
        rack_tiles = []
        for tile_data in rack_data:
            if tile_data is not None:
                rack_tiles.append(Tile(
                    letter=tile_data.get('letter', ''),
                    is_blank=tile_data.get('isBlank', False)
                ))

        board.set_board(processed_board)
        top_moves = [m._asdict() for m in board.get_all_legal_moves(rack_tiles)][:10]

        return jsonify({
            "status": "success",
            "message": "Board and rack data received",
            "top_moves": top_moves
        }), 200

    except KeyError as e:
        return jsonify({"error": f"Missing required field: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=8085, use_reloader=False)