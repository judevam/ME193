#include <Arduino_LED_Matrix.h>
#include <Arduino_RouterBridge.h>

const int MATRIX_ROWS = 8;
const int MATRIX_COLS = 13;

Arduino_LED_Matrix matrix;

void show_dot(int col, int row);
void clear_matrix();

void setup() {
  matrix.begin();
  matrix.setGrayscaleBits(3);  // frame values are 0..7
  matrix.clear();

  Bridge.begin();
  Bridge.provide("show_dot", show_dot);
  Bridge.provide("clear_matrix", clear_matrix);
}

void loop() {}

void show_dot(int col, int row) {
  uint8_t frame[MATRIX_ROWS * MATRIX_COLS] = {0};
  if (col >= 0 && col < MATRIX_COLS && row >= 0 && row < MATRIX_ROWS) {
    frame[row * MATRIX_COLS + col] = 7;  // max brightness (3-bit grayscale)
  }
  matrix.draw(frame);
}

void clear_matrix() {
  matrix.clear();
}
