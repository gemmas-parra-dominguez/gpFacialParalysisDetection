#include "gpmisc.h"

void writeCSV(string filename, Mat m)
{
       ofstream myfile;
       myfile.open(filename.c_str());
       myfile<< format(m, cv::Formatter::FMT_CSV) << endl;
       myfile.close();
}

Mat readCSV(string file_name)
{
    ifstream inputfile(file_name);
    string current_line;
    // vector allows you to add data without knowing the exact size beforehand
    vector< vector<int> > all_data;
    // Start reading lines as long as there are lines in the file
    while(getline(inputfile, current_line))
    {
       // Now inside each line we need to seperate the cols
       vector<int> values;
       stringstream temp(current_line);
       string single_value;
       while(getline(temp,single_value,',')){
            // convert the string element to a integer value
            values.push_back(atoi(single_value.c_str()));
       }
       // add the row to the complete data vector
       all_data.push_back(values);
    }
    // Now add all the data into a Mat element
    Mat landmarks_vec = Mat::zeros((int)all_data.size(), (int)all_data[0].size(), CV_32FC1);
    // Loop over vectors and add the data
    for(int rows = 0; rows < (int)all_data.size(); rows++)
    {
       for(int cols= 0; cols< (int)all_data[0].size(); cols++)
        {
          landmarks_vec.at<float>(rows,cols) = all_data[rows][cols];
       }
    }

	return landmarks_vec;
}
