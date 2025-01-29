import plotly.graph_objects as go
import io
import base64

def test_kaleido():
    fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
    buf = io.BytesIO()
    try:
        fig.write_image(buf, format='png')
        encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
        print("Kaleido is working properly.")
    except Exception as e:
        print(f"Error with Kaleido: {e}")

test_kaleido()
