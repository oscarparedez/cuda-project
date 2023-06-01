#pragma once

class PGMImage
{
public:
	PGMImage(char*);
	PGMImage(int x, int y, int col);
	~PGMImage();
	bool write(char*);

	int x_dim;
	int y_dim;
	int num_colors;
	unsigned char* pixels;
};
