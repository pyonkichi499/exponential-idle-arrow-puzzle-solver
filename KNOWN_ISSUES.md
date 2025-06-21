# Known Issues and Limitations

## Image Recognition (Auto-solve feature)

### Current Status
The image recognition feature (`auto-solve` command) is currently a **proof of concept** and has not been tested with actual Exponential Idle gameplay.

### Limitations

1. **Digit Recognition**
   - Uses a simple pixel ratio heuristic that is NOT calibrated for actual game graphics
   - The thresholds in `vision.py` are placeholder values
   - Will likely fail to correctly recognize digits in the actual game

2. **Grid Detection**
   - The contour detection may not work with all game resolutions or themes
   - Assumes a square grid with clear boundaries
   - No handling for partially visible puzzles or UI overlays

3. **Missing Features**
   - No template matching with actual game assets
   - No OCR implementation
   - No machine learning model for robust digit recognition
   - No handling of different game themes or color schemes

### What Needs to Be Done

To make the auto-solve feature production-ready:

1. **Capture Game Assets**
   - Take screenshots of each digit (0-4) from the actual game
   - Capture samples at different resolutions
   - Account for different themes if the game has them

2. **Implement Proper Recognition**
   - Replace the pixel ratio method with:
     - Template matching using actual digit images
     - OCR using pytesseract
     - Or train a small CNN for digit recognition

3. **Improve Grid Detection**
   - Test with various game resolutions
   - Handle edge cases (partial visibility, UI overlays)
   - Add visual feedback for detected grid boundaries

4. **Calibration Tool**
   - Create a tool to help users calibrate recognition for their setup
   - Save calibration data for future use

### Workaround

Until the image recognition is properly implemented, users can:
- Use the manual input methods (`solve` command with text input)
- Contribute actual game screenshots to help with development
- Implement their own recognition methods by extending the `PuzzleDetector` class

## Other Known Issues

### Algorithm Limitations
- The demo puzzle in `cli.py` currently fails to solve, suggesting the algorithm may not handle all puzzle configurations
- No guarantee that all possible puzzles can be solved with the current implementation

### Performance
- Large puzzles (>7x7) have not been extensively tested
- No optimization for solving multiple puzzles in sequence