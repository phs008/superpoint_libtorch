#include <opencv2/core/mat.hpp>
#include "SPextractor.h"

#include "sys/types.h"
#include "dirent.h"
#include "stdlib.h"

//#include <conio.h>
#include <iostream>
#include <thread>

using namespace std;
using namespace SuperPoint;

int main()
{
    string folderName = "./sample/";
    struct dirent **entry;
    int n = scandir(folderName.c_str(), &entry, 0, alphasort);
    SPextractor spExtractor = SPextractor("superpoint.pt", 0.015, true, 8, 0.5, true);
    int i = 0;
    while (i < n)
    {

        if (entry[i]->d_type != 4)
        {
            string fileName = entry[i]->d_name;
            string fullPath;
            fullPath = folderName + fileName;
            cout << fullPath << endl;
            cv::Mat img = cv::imread(fullPath, cv::IMREAD_GRAYSCALE);
            img.convertTo(img, CV_32F, 1.0f / 255.0f);
            /// SuperPoint Run function
            auto matchingPoint = spExtractor.Run(img, fullPath);
//            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        free(entry[i]);
        ++i;
    }
}