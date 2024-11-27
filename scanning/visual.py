def generate_scrabble_board_html(matrix):
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            table {
                border-collapse: collapse;
                width: 100%;
                max-width: 600px;
                margin: 20px auto;
                font-family: Arial, sans-serif;
            }
            td {
                border: 1px solid black;
                width: 40px;
                height: 40px;
                text-align: center;
                vertical-align: middle;
                font-size: 20px;
                font-weight: bold;
                background-color: #f4f4f4;
            }
            td input {
                width: 100%;
                height: 100%;
                border: none;
                text-align: center;
                font-size: 20px;
                font-weight: bold;
                background: none;
            }
        </style>
    </head>
    <body>
        <table>
    """
    for row in matrix:
        html += "<tr>\n"
        for cell in row:
            value = cell if cell is not None else ""
            html += f'<td><input type="text" maxlength="1" value="{value}"></td>\n'
        html += "</tr>\n"
    html += """
        </table>
    </body>
    </html>
    """
    return html

def save(board):
    with open("scrabble_board.html", "w") as f:
        html_content = generate_scrabble_board_html(board)
        f.write(html_content)

