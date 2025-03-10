from music21 import converter, note, configure, stream
import os

def render_kern():
    kern_file = os.path.join("data", "test", "Rue1024a.krn")
    # kern_file = os.path.join("data", "full", "more_than_10", "SELECTED", "Oke", "Oke1004.1c-Missa_Cuiusvis_toni-Credo-on_D.krn")
    kern_file = os.path.join("data", "test", "extra_parFifth_rue1024a.krn")
    # 
    try:
        score = converter.parse(kern_file)
        score.show()
    except Exception as original_exception: 
        print(original_exception)          
        try:
            # Parse with specific Humdrum converter options
            original_score = converter.parse(kern_file, format='humdrum', makeNotation=False)
            
            # Create an entirely new score
            new_score = stream.Score()
            
            # Process each part individually
            for i, original_part in enumerate(original_score.parts):
                # Create a new part
                new_part = stream.Part(id=f'reconstructed_{i}')
                
                # Copy essential part metadata
                if original_part.id:
                    new_part.id = original_part.id
                
                # Create a dictionary to track measures by number
                measures_by_number = {}
                
                # First pass - identify and collect measures
                for measure in original_part.getElementsByClass('Measure'):
                    if measure.number not in measures_by_number:
                        measures_by_number[measure.number] = []
                    measures_by_number[measure.number].append(measure)
                
                # Second pass - use only the first instance of each measure number
                for measure_number in sorted(measures_by_number.keys()):
                    original_measure = measures_by_number[measure_number][0]
                    
                    # Create a new measure with the same number
                    new_measure = stream.Measure(number=measure_number)
                    
                    # Copy the contents (notes, rests, etc.)
                    for element in original_measure.elements:
                        if isinstance(element, (note.Note, note.Rest, note.Chord)):
                            new_measure.insert(element.offset, copy.deepcopy(element))
                    
                    # Add the measure to the new part
                    new_part.append(new_measure)
                
                # Add the new part to the score
                new_score.insert(0, new_part)
            
            # Apply notation to our cleaned structure
            cleaned_score = new_score.makeNotation(inPlace=False)

            # Render the score
            cleaned_score.show()
            
        except Exception as e:
            print("Error parsing the kern file. Original error", original_exception)
            raise(e)
   
    return



if __name__ == '__main__':
    render_kern()
    # configure.run()
