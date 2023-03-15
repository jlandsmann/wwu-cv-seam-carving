// SeamCarving.cpp : Diese Datei enthält die Funktion "main". Hier beginnt und endet die Ausführung des Programms.
//
#include "SeamCarving.hpp"

static const string destImagePath = ""; //ToEdit
static const string sourceImagePath = "";
static const int IMAGESIZE = 200;
int main(){
    fstream metadaten(destImagePath+"metadaten.csv", ios::out|std::ios::app);
    if (!metadaten.good()) {
        metadaten.close();
        cout << "Filestream konnte nicht geöffnet werden" << endl;
        return 0;
    }
    srand((unsigned)time(NULL));
    double vertical=0.0, horizontal=0.0;
   for (auto const& dir_entry : filesystem::directory_iterator(sourceImagePath)) {
        if (dir_entry.is_directory()) continue;
        Mat img = imread(dir_entry.path().string());
        Mat resized;
        if (img.type() != 16) {
            cout << "Falscher Bildtyp bei: " << dir_entry.path().string() << endl;
            continue;
        }
        bool VoH = (img.rows < img.cols);
        double ratio;
        if (VoH) {
            resize(img, resized, Size((int)(img.cols* (double)IMAGESIZE /img.rows), IMAGESIZE), INTER_AREA);
        }
        else {
            resize(img, resized, Size(IMAGESIZE,(int)(img.rows * (double)IMAGESIZE / img.cols)), INTER_AREA);
        }
        imwrite(destImagePath + "resized\\SCBild" + regex_replace(dir_entry.path().filename().string(), regex("(SCBild \\()|(\\).(jpg|png))"), "") + "_resized.jpg",resized);
        int carveOrCrop = rand() % 10;
        if (carveOrCrop < 7) {
            ratio = (VoH ? (double)IMAGESIZE / resized.cols : (double)IMAGESIZE / resized.rows);
            img = seamCarving(resized, IMAGESIZE, IMAGESIZE);
        } else {
            img = resized(Range(0, IMAGESIZE), Range(0, IMAGESIZE));
            ratio = 1.;
        }
        metadaten << "SCBild" + regex_replace(dir_entry.path().filename().string(), regex("(SCBild \\()|(\\).(jpg|png))"), "") + ".jpg" << ";";
        metadaten << regex_replace(dir_entry.path().filename().string(), regex("(SCBild \\()|(\\).(jpg|png))"), "") << ";";
        metadaten << (carveOrCrop < 7) << ";";
        metadaten << ratio << ";";
        metadaten << (VoH?"v" : "h") << endl;
        imwrite(destImagePath + "SCBild" + regex_replace(dir_entry.path().filename().string(), regex("(SCBild \\()|(\\).(jpg|png))"), "") + ".jpg", img);
        cout << dir_entry.path().string() <<" wurde gecarved" << endl;
    }
    metadaten.close();
    return 0;
}


Mat seamCarving(Mat img,const double ratioCols,const double ratioRows){
    return seamCarving(img, (int)(ratioCols * img.cols), (int)(ratioRows * img.rows));
}

Mat seamCarving(Mat img,int newCols,int newRows){
#ifdef _DEBUG
    imshow("Image to Seamcarve", img);
    waitKey(0);
#endif
    if (newCols&&newCols<img.cols && newCols> 0) {
        img=carve(img, newCols);
    }
    if (newRows && newRows<img.rows && newRows> 0) {
        Mat transposeImg;
        transpose(img, transposeImg);
        transposeImg = carve(transposeImg, newRows);
        transpose(transposeImg, img);
    }
    
    return img;
}

Mat carve(Mat img, int newCols){
    Mat energyImg = energyImage(img);
#ifdef _DEBUG
    Mat seamImg = img.clone();
    Mat energyImgShow(img.rows, img.cols, CV_8UC1);
    energyImg.convertTo(energyImgShow, CV_8U, 255.0 / 1000); //Faktor 1000 geraten für gute Skalierung
    imshow("Energie", energyImgShow);
    waitKey(0);
    vector<pair<int, int>> markiertePixel;
#endif
    // Wiederhole das Entfernen von Seams, bis die gewünschte Breite erreicht ist
    while (img.cols > newCols) {
        // Finde den Energiepfad mit dem geringsten Energieverbrauch
        vector<int> seam = findSeam(energyImg);
        // Entferne den gefundenen Pfad aus dem Bild
        Mat img2(img.rows, img.cols - 1, CV_8UC3);
        Mat energyImg2(img.rows, img.cols - 1, CV_32F);
        int k = 0;
        for (int x = 0; x < img.rows; x++) {
            int ynew = 0;
            for (int y = 0; y < img.cols; y++) {
                if (y != seam[x]) {
                    img2.at<Vec3b>(x, ynew) = img.at<Vec3b>(x, y);
                    energyImg2.at<float>(x, ynew) = energyImg.at<float>(x, y);
                    ynew++;
                }
            }
        }
        img = img2;
        energyImage(img, energyImg2, seam);
        energyImg = energyImg2;
        
#ifdef _DEBUG
        for (int i = 0; i < img.rows; i++) {
            markiertePixel.push_back(pair<int, int>(i, seam[i]));
        }
#endif
    }
#ifdef _DEBUG
    for (pair<int, int> Pixel : markiertePixel) {
        seamImg.at<Vec3b>(Pixel.first, Pixel.second) = Vec3b(0,0,255);
    }
    imwrite("Seambild.jpg", seamImg);
#endif
    return img;
}

// Funktion zum Finden des Energiepfads einer gegebenen Spalte
vector<int> findSeam(Mat& energy) {
    int rows = energy.rows;
    int cols = energy.cols;

    // Matrix für die Entfernungen
    Mat dist(rows, cols, CV_32F);
    dist.setTo(numeric_limits<float>::max());
    energy.row(0).copyTo(dist.row(0));
    // Matrix für die Vorgänger des Energiepfads
    Mat prev(rows, cols, CV_32S);
    for (int j = 0; j < cols; j++) {
            prev.at<int>(0, j) = j;
    }
    // Dynamic Programming, um den Energiepfad zu finden
    for (int i = 1; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dist.at<float>(i, j) = dist.at<float>(i - 1, j) + energy.at<float>(i, j);
            prev.at<int>(i, j) = j;
            if (j > 0) {
                if (dist.at<float>(i, j) > dist.at<float>(i - 1, j - 1) + energy.at<float>(i, j)) {
                    dist.at<float>(i, j) = dist.at<float>(i - 1, j - 1) + energy.at<float>(i, j);
                    prev.at<int>(i, j) = j - 1;
                }
            }
            if (j < cols - 1) {
                if (dist.at<float>(i, j) > dist.at<float>(i - 1, j + 1) + energy.at<float>(i, j)) {
                    dist.at<float>(i, j) = dist.at<float>(i - 1, j + 1) + energy.at<float>(i, j);
                    prev.at<int>(i, j) = j + 1;
                }
            }
        }
    }

    // Finden des minimalen Energiepfads
    int minRow = 0;
    float minDist = dist.at<float>(rows - 1, 0);
    for (int j = 1; j < cols; j++) {
        if (dist.at<float>(rows - 1, j) < minDist) {
            minDist = dist.at<float>(rows - 1, j);
            minRow = j;
        }
    }

    // Erstellen des Energiepfads
    vector<int> seam(rows);
    seam[rows - 1] = minRow;
    for (int i = rows - 2; i >= 0; i--) {
        seam[i] = prev.at<int>(i + 1, seam[i + 1]);
    }

    return seam;
}

//Erstellen eines komplett neuen Energiebilds
Mat energyImage(Mat& img) {
    //Bildtypenunterscheidung
    Mat grayimg(img.rows, img.cols, CV_8UC1);
    Mat grayimgborder(img.rows + 2, img.cols + 2, CV_8UC1);
    cvtColor(img, grayimg, cv::COLOR_RGB2GRAY);
    copyMakeBorder(grayimg, grayimgborder, 1, 1, 1, 1, BORDER_REPLICATE);
    Mat energyImg(img.rows, img.cols, CV_32F);
    for (int x = 0; x < img.rows; x++) {
        for (int y = 0; y < img.cols; y++) {
            energyImg.at<float>(x, y) = energy(grayimgborder, x + 1, y + 1);
        }
    }
    return energyImg;
}

//Altualisieren eines bestehenden Energiebilds
void energyImage(Mat& img, Mat& oldEnergyImg, vector<int>& seam){
    Mat grayimg(img.rows, img.cols, CV_8UC1);
    Mat grayimgborder(img.rows + 2, img.cols + 2, CV_8UC1);
    cvtColor(img, grayimg, cv::COLOR_RGB2GRAY);
    copyMakeBorder(grayimg, grayimgborder, 1, 1, 1, 1, BORDER_REPLICATE);
    for (int x = 0; x < img.rows;++x) {
        for (int y = max(0, seam[x] - 2); y < min(img.cols, seam[x] + 2); y++) {
            oldEnergyImg.at<float>(x,y)= energy(grayimgborder, x + 1, y + 1);
        }
    }
}

//Energieberechnung eines Pixel mittels des Sobel-Operators
float energy(Mat& img, int x, int y) {
    float gradX = (float)img.at<uchar>(x + 1, y+1) - (float)img.at<uchar>(x - 1, y+1);
    gradX += 2.f*img.at<uchar>(x + 1, y) - 2.f*img.at<uchar>(x - 1, y);
    gradX += (float)img.at<uchar>(x + 1, y-1) - (float)img.at<uchar>(x - 1, y-1);

    float gradY = (float)img.at<uchar>(x + 1, y + 1) - (float)img.at<uchar>(x + 1, y-1);
    gradY += 2.f * img.at<uchar>(x , y+1) - 2.f * img.at<uchar>(x , y-1);
    gradY += (float)img.at<uchar>(x - 1, y +1) - (float)img.at<uchar>(x - 1, y - 1);

    return sqrt(gradX * gradX + gradY * gradY);
}
