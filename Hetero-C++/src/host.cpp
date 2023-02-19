#include "host.h"
#include "hd.h"

using namespace std;

void datasetBinaryRead(vector<int> &data, string path){
	ifstream file_(path, ios::in | ios::binary);
	assert(file_.is_open() && "Couldn't open file!");
	int32_t size;
	file_.read((char*)&size, sizeof(size));
	int32_t temp;
	for(int i = 0; i < size; i++){
		file_.read((char*)&temp, sizeof(temp));
		data.push_back(temp);
	}
	file_.close();
}

int main(int argc, char** argv)
{
	auto t_start = chrono::high_resolution_clock::now();
   
	vector<int> X_train;
	vector<int> y_train;
	
	datasetBinaryRead(X_train, X_train_path);
	datasetBinaryRead(y_train, y_train_path);

	int N_SAMPLE = y_train.size();
	int input_int = X_train.size();
	 
	vector<int, aligned_allocator<int>> input_gmem(input_int);
	for(int i = 0; i < input_int; i++){
		input_gmem[i] = X_train[i];
	}
	
	vector<int, aligned_allocator<int>> labels_gmem(N_SAMPLE);
	for(int i = 0; i < N_SAMPLE; i++){
		labels_gmem[i] = y_train[i];
	}

	//We need a seed ID. To generate in a random yet determenistic (for later debug purposes) fashion, we use bits of log2 as some random stuff.
	vector<int, aligned_allocator<int>> ID_gmem(Dhv/32);
	srand (time(NULL));
	for(int i = 0; i < Dhv/32; i++){
		long double temp = log2(i+2.5) * pow(2, 31);
		long long int temp2 = (long long int)(temp);
		temp2 = temp2 % int(pow(2, 31));
		ID_gmem[i] = int(temp2);
		//ID_gmem[i] = int(rand());
	}
	vector<int, aligned_allocator<int>> classHV_gmem(N_CLASS*Dhv);	
	vector<int, aligned_allocator<int>> trainScore(1);
	
	vector<HyperVector512, aligned_allocator<HyperVector512>> encHV_gmem((Dhv/32)*N_SAMPLE*sizeof(HyperVector512)/sizeof(int));

	auto t_elapsed = chrono::high_resolution_clock::now() - t_start;
	long mSec = chrono::duration_cast<chrono::milliseconds>(t_elapsed).count();
	long mSec_train = mSec;

	auto buf_input = input_gmem.data();
	auto buf_ID = ID_gmem.data();
	auto buf_classHV = classHV_gmem.data();
	auto buf_labels = labels_gmem.data();
	auto buf_encHV = encHV_gmem.data();
	auto buf_trainScore = trainScore.data();
	cout << "Training with " << N_SAMPLE << " samples." << endl;

	t_start = chrono::high_resolution_clock::now();
	hd(buf_input,
	   buf_ID,
	   buf_classHV,
	   buf_labels,
	   buf_encHV,
	   buf_trainScore,
	   train,
	   N_SAMPLE);
	t_elapsed = chrono::high_resolution_clock::now() - t_start;
	
	mSec = chrono::duration_cast<chrono::milliseconds>(t_elapsed).count();
	cout << "Reading train data took " << mSec_train << " mSec" << endl;
	cout << "Train execution took " << mSec << " mSec" << endl;
	
	/*for(int i = 0; i < N_CLASS; i++){
		cout << classHV_gmem[i*Dhv] << "\t" << classHV_gmem[i*Dhv + Dhv - 1] << endl;
	}*/
	cout << "Train accuracy = " << float(trainScore[0])/N_SAMPLE << endl << endl;
	
	t_start = chrono::high_resolution_clock::now();
	vector<int> X_test;
	vector<int> y_test;
	
	datasetBinaryRead(X_test, X_test_path);
	datasetBinaryRead(y_test, y_test_path);

	input_int = X_test.size();	 
	input_gmem.resize(input_int);
	for(int i = 0; i < input_int; i++){
		input_gmem[i] = X_test[i];
	}
	
	t_elapsed = chrono::high_resolution_clock::now() - t_start;
	mSec = chrono::duration_cast<chrono::milliseconds>(t_elapsed).count();
	long mSec_test = mSec;
	
	int N_TEST = y_test.size();
	labels_gmem.resize(N_TEST);
	
	auto buf_input2 = input_gmem.data();
	auto buf_labels2 = labels_gmem.data();
    	train = 0; //i.e., inference

	t_start = chrono::high_resolution_clock::now();
	hd(buf_input2,
	   buf_ID,
	   buf_classHV,
	   buf_labels2,
	   buf_encHV,
	   buf_trainScore,
	   train,
	   N_SAMPLE);
    	t_elapsed = chrono::high_resolution_clock::now() - t_start;

    	mSec = chrono::duration_cast<chrono::milliseconds>(t_elapsed).count();
    	cout << "Reading test data took " << mSec_test << " mSec" << endl;
	cout << "Test execution took " << mSec << " mSec" << endl;
    
    	int correct = 0;
    	for(int i = 0; i < N_TEST; i++)
    		if(labels_gmem[i] == y_test[i])
    			correct += 1;
    	cout << "Test accuracy = " << float(correct)/N_TEST << endl;
}

