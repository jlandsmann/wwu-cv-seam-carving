// SeamCarving.cpp : Diese Datei enthält die Funktion "main". Hier beginnt und endet die Ausführung des Programms.
//
#include "SeamCarving.hpp"

int main() {
    //Beispielprogramm 
    Mat img = imread("Testbild.jpg");
    if (img.empty()) return -1;
    Mat ausgabebild = seamCarving(img, 1550, 1068);
    imwrite("Testbild_gecarved.jpg", ausgabebild);
}

//Seamcarving mit keiner konkreten neuen Größenangabe für das Bild, sondern mit Prozentzahlen für die einzelnen Bildseiten
Mat seamCarving(Mat img,const double ratioCols,const double ratioRows){
    return seamCarving(img, (int)(ratioCols * img.cols), (int)(ratioRows * img.rows));
}
//Seamcarving mit neuen Größenangaben für beides Seiten des Bildes
Mat seamCarving(Mat img,int newCols,int newRows){

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

//Vereinfachte Seamcarvingmethode, die nur die Spalten des Bildes reduziert
Mat carve(Mat img, int newCols){
    Mat energyImg = energyImage(img);
    // Wiederhole das Entfernen von Seams, bis die gewünschte Breite erreicht ist
    while (img.cols > newCols) {
        // Finde den Energiepfad mit dem geringsten Energiesumme
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
    }
    return img;
}

// Funktion zum Finden des Energiepfads mit der geringsten Energiesumme
vector<int> findSeam(Mat& energy) {
    int rows = energy.rows;
    int cols = energy.cols;

    // Matrix für die Energie
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
    //Vorbereitungen für die Energieberechnung
    Mat grayimg(img.rows, img.cols, CV_8UC1);
    Mat grayimgborder(img.rows + 2, img.cols + 2, CV_8UC1);
    cvtColor(img, grayimg, cv::COLOR_RGB2GRAY);
    copyMakeBorder(grayimg, grayimgborder, 1, 1, 1, 1, BORDER_REPLICATE);
    Mat energyImg(img.rows, img.cols, CV_32F);

    //Energieberechnung für jeden Pixel
    for (int x = 0; x < img.rows; x++) {
        for (int y = 0; y < img.cols; y++) {
            energyImg.at<float>(x, y) = energy(grayimgborder, x + 1, y + 1);
        }
    }
    return energyImg;
}

//Altualisieren eines bestehenden Energiebilds
void energyImage(Mat& img, Mat& oldEnergyImg, vector<int>& seam){
    //Vorbereitungen für die Energieberechnung
    Mat grayimg(img.rows, img.cols, CV_8UC1);
    Mat grayimgborder(img.rows + 2, img.cols + 2, CV_8UC1);
    cvtColor(img, grayimg, cv::COLOR_RGB2GRAY);
    copyMakeBorder(grayimg, grayimgborder, 1, 1, 1, 1, BORDER_REPLICATE);

    //Neuberechnung für die Pixel bei denen sich potentiell die Energie geändert hat
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

    return abs(gradX) + abs(gradY);
}

//Methode für die Datenvorbereitung wie in Abschnitt 2.2 in Ausarbeitung beschrieben
//Skaliert alle Bilder aus dem sourceImagePath auf ImageSize x ImgageSize und speichert es im destImagePath
//in der Ausarbeitung war Imagesize 200
void data(string destImagePath, string sourceImagePath, int imageSize) {

    fstream metadaten(destImagePath + "metadaten.csv", ios::out | std::ios::app);
    if (!metadaten.good()) {
        metadaten.close();
        cout << "Filestream konnte nicht geöffnet werden" << endl;
        return;
    }

    srand((unsigned)time(NULL));
    double vertical = 0.0, horizontal = 0.0;
    //Skaliere jedes Bild im Sourceordner neu
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
        //Die kürzere Seite wird auf Imagesize skaliert 
        if (VoH) {
            resize(img, resized, Size((int)(img.cols * (double)imageSize / img.rows), imageSize), INTER_AREA);
        }
        else {
            resize(img, resized, Size(imageSize, (int)(img.rows * (double)imageSize / img.cols)), INTER_AREA);
        }

        //Es wird zufällig entschieden, ob ein Bild geseamcarvt wird oder einfach abgeschnitten wird
        int carveOrCrop = rand() % 10;
        if (carveOrCrop < 7) {
            //Fall1: Bild wird geseamcarvt. Ratio gibt an wie stark das Bild geseamcarvt wurde
            ratio = (VoH ? (double)imageSize / resized.cols : (double)imageSize / resized.rows);
            img = seamCarving(resized, imageSize, imageSize);
        }
        else {
            //Fall2: Einfaches Abschneiden des Bildes zu der Größe ImageSize x ImageSize
            img = resized(Range(0, imageSize), Range(0, imageSize));
            ratio = 1.;
        }

        //Metadaten Datei wird mit Werten gefüllt. Diese Datei wird verwendet um die Bilder für das Training mit Labels zu versehen
        metadaten << "SCBild" + regex_replace(dir_entry.path().filename().string(), regex("(SCBild \\()|(\\).(jpg|png))"), "") + ".jpg" << ";";
        metadaten << regex_replace(dir_entry.path().filename().string(), regex("(SCBild \\()|(\\).(jpg|png))"), "") << ";";
        metadaten << (carveOrCrop < 7) << ";";
        metadaten << ratio << ";";
        metadaten << (VoH ? "v" : "h") << endl;
        imwrite(destImagePath + "SCBild" + regex_replace(dir_entry.path().filename().string(), regex("(SCBild \\()|(\\).(jpg|png))"), "") + ".jpg", img);
        cout << dir_entry.path().string() << " wurde gecarved" << endl;
    }
    metadaten.close();
    return;
}
