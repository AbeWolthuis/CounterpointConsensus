from music21 import converter, configure
import os

def render_kern():
    kern_file = os.path.join("data", "test", "Rue1024a.krn")
    kern_file = os.path.join("data", "full", "more_than_10", "SELECTED", "Oke", "Oke1004.1c-Missa_Cuiusvis_toni-Credo-on_D.krn")
    # kern_file = os.path.join("data", "test", "extra_parFifth_rue1024a.krn")
    # 
    try:
        score = converter.parse(kern_file)
    except Exception as e:
        print("Error parsing the kern file:", e)
        return

    # Render the score. This will open the default music notation viewer
    score.show()

if __name__ == '__main__':
    render_kern()
    # configure.run()
