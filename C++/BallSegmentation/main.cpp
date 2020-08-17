#include "utils.h"

int main()
{
	String imgFolder = "../Images/";
	String saveFolder = "../Images/Save/";

	ImageSegmentation session(imgFolder, saveFolder);
	session.drawMask();

	return 0;
}