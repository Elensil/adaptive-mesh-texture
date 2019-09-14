#include "../Include/my_mesh_encoder.h"

cv::Mat MeshEncoder::getTriangularDCTMatrix(const int N, std::vector<int> &quantizationMat)const{
    int S = N*(N+1)/2;    //number of samples. (i.e. matrix is of size S*S)
                                            //DCT block is of size N*(N+1)

    quantizationMat.clear();
    quantizationMat.reserve(S);
    //We need to choose a writing order for DCT coefficients C[p,q]
    //arbitrary choice:
    //if p1+q1<p2+q2, (p1,q1) < (p2,q2)
    //if p1+q1==p2+q2, if q1<q2, (p1,q1) < (p2,q2)
    //e.g. (0,0), (0,2), (1,1), (2,0), (0,4), (1,3), (3,1), (4,0), ...
    cv::Mat dctMat (S,S,CV_32FC1,cv::Scalar(0.0f));      //32 or 64 bits? I suppose 32 is enough

    int m=0, n=0, p=0, q=0;     //(p,q): DCT coefficients indices.
                                //(m,n): triangle samples indices.
    float kp0 = sqrt(1.0f/N);
    float hq0 = sqrt(1.0f/(N+1));
    float kp1 = sqrt(2.0f/N);
    float hq1 = sqrt(2.0f/(N+1));   //N corresponds to M in the paper. (N+1) is N in the paper
    float kp, hq;

    for(int i=0;i<S;++i)            //line index, i.e. DCT
    {

        if(p==0)
        {
            kp = kp0;
        }else{
            kp = kp1;
        }
        if(q==0)
        {
            hq = hq0;
        }else{
            hq = hq1;
        }
        m=0;
        n=0;
        for(int j=0; j<S; ++j)      //column index, i.e. color samples
        {
            //log(ALWAYS)<<"(p,q)=("<<p<<","<<q<<"), (m,n)=("<<m<<","<<n<<")"<<endLog();
            
            //Just copied equation from paper
            dctMat.at<float>(i,j) = sqrt(2)*kp*hq*cos(M_PI*p*(float(m)+0.5)/N)*cos(M_PI*q*(float(n)+0.5)/(N+1));
            //log(ALWAYS)<<"1st cos = "<<cos(M_PI*q*(float(n)+0.5)/(N+1))<<endLog();
            //log(ALWAYS)<<"2nd cos = "<<cos(M_PI*q*(float(m)+0.5)/N)<<endLog();
            //log(ALWAYS)<<"kp = "<<kp<<endLog();
            //log(ALWAYS)<<"hq = "<<hq<<endLog();
            //log(ALWAYS)<<"value: "<<dctMat.at<float>(i,j)<<endLog();
            quantizationMat[i] = int(1+1.5*p+1.5*q+p*q); //test
            if(m+n==N-1)
            {
                m+=1;
                n=0;
            }
            else
            {
                n+=1;
            }

        }   //end column loop

        //update p,q,m,n. This is the tricky part
        q-=1;
        p+=1;
        if((q<0)||(p>N-1))      //we reached the end of the 'diagonal'
        {
            q = p+q+2;          //go to next diagonal
            p = 0;
        }
        /*if((p+q)>(2*N-1))     //shouldn't happen
        {
            log(ALWAYS)<<"WTFFFFFFFFFFFFFFFFF!!"<<endLog();
            return dctMat;
        }*/
        while(q>N)          //reach the beginning of the diagonal
        {
            q-=1;
            p+=1;
        }
    }   //end line loop

    return dctMat;
}



void MeshEncoder::writeJPEGMeshColor(std::map<int,cv::Mat> &dctTrianglesList, std::map<int,std::vector<float>> &quantizationTables, std::map<int,cv::Mat> &eigenVectors, std::string filePath, std::string fileName)
{
    //open file to write in
    std::ofstream fout(filePath+fileName, std::ios::out | std::ios::binary);
    


    BitArray myBitArray(5000000);

    //myBitArray.byteArray = byteArray;
    //myBitArray.leftoverBits = 0;

    //std::vector<char> myBitArray = byteArray;

    //write header/metadata
    //Start of Image
    myBitArray.writeBytes(m_SOI);

    //Quantization tables
    

    //write data
    std::map<int,cv::Mat>::iterator it;
    int resNumber=0;
    //loop over resolution levels
    for(it = dctTrianglesList.begin();it!=dctTrianglesList.end();++it)
    {

        int myRes = it->first;
        cv::Mat patterns = it->second;
        //int sampNum = 3*(myRes-1)*(myRes-2)/2;
        //int  sampNum = (myRes+2)*(myRes+1)/2 + (myRes/2+2)*(myRes/2+1);
        int sampNum = patterns.rows;

        //create huffman trees
        HuffTree dcYTree = HuffTree();
        HuffTree acYTree = HuffTree();

        int patternsNum = patterns.cols;

        int lastDcCoef=0;
        int trailingZeros=0;

        //first loop to constitute a list of coefficient for huffman tables
        for(int i=0;i<patternsNum;++i)      
        {
            for(int j=0;j<sampNum;++j)
            {
                if(j==0)        //DC coef
                {
                    addDCCoef(int(patterns.at<float>(j,i))-lastDcCoef, dcYTree);
                    lastDcCoef = int(patterns.at<float>(j,i));
                }
                else                        //AC coef
                {
                    if(int(patterns.at<float>(j,i))==0)
                    {
                        if(j==(sampNum-1))        //end of block, no need to write all the zeros
                        {
                            acYTree.addOccurences((unsigned char)0,1);
                            trailingZeros=0;
                        }
                        else
                        {
                            ++trailingZeros;
                        }
                    }
                    else
                    {
                        addACCoef(int(patterns.at<float>(j,i)), trailingZeros, acYTree);
                        trailingZeros=0;
                    }
                }
            }

        }

        //Compute Huffman tables
        dcYTree.computeHuffmanTable();
        acYTree.computeHuffmanTable();

        //Writing resolution change
        myBitArray.writeBytes(m_APP4);
        myBitArray.writeBytes(m_APP4);
        myBitArray.writeSingleByte(myRes);

        //Writing Huffman tables
        writeHuffTree(myBitArray, dcYTree);
        writeHuffTree(myBitArray, acYTree);

        //Writing Quantization tables
        encodeQuantizationMatrix(myBitArray, quantizationTables[myRes], resNumber);

        //Writing eigen vectors
        encodeEigenVectors(myBitArray, eigenVectors[myRes], sampNum+1, resNumber);  //+1 for mean vector
        //log(ALWAYS)<<"Tables written!"<<endLog();

        //Start of Scan
        myBitArray.writeBytes(m_SOS);
        //recompute everything
        lastDcCoef=0;
        trailingZeros=0;

        for(int i=0;i<patternsNum;++i)      
        {
            for(int j=0;j<sampNum;++j)
            {

                if(myBitArray.getByteSize()>80000000)
                {
		    log(ALWAYS)<<"size exceeded!(?)"<<endLog();
                    goto writingFile;
                }
                if(j==0)        //DC coef
                {
                    
                    writeCodedDCCoef(myBitArray, int(patterns.at<float>(j,i))-lastDcCoef, dcYTree);    //needs to be per channel?
                    lastDcCoef = int(patterns.at<float>(j,i));
                }
                else                        //AC coef
                {
                    if(int(patterns.at<float>(j,i))==0)
                    {
                        if(j==(sampNum-1))        //end of block, no need to write all the zeros
                        {
                            //log(ALWAYS)<<"writing EOB"<<endLog();
                            //myBitArray.writeBytes(m_EOB);
                            myBitArray.writeBits(acYTree.getCodedByte(0),acYTree.getBitNum(0));
                            trailingZeros=0;
                        }
                        else
                        {
                            ++trailingZeros;
                        }
                    }
                    else
                    {
                        writeCodedACCoef(myBitArray, int(patterns.at<float>(j,i)), trailingZeros, acYTree);
                        trailingZeros=0;
                    }
                }
            }

        }
        ++resNumber; 	//used to index quantization tables (at least)
        myBitArray.padLastByte();

    }
    writingFile:
    //log(ALWAYS)<<"myBitArray size: "<<myBitArray.getByteSize()<<endLog();
    myBitArray.padLastByte();
    std::vector<unsigned char> byteArray;
    byteArray = myBitArray.getWrittenBytes();

    log(ALWAYS)<<"myBitArray size: "<<myBitArray.getByteSize()<<endLog();
    // log(ALWAYS)<<"byteArray size: "<<byteArray.size()<<" ("<<sizeof(byteArray[0])<<")"<<endLog();
    // log(ALWAYS)<<"remaining bytes: "<<myBitArray.getRemainingBytesNumber()<<endLog();
    fout.write((char*)&byteArray[0],byteArray.size());
    
    //close file
    fout.close();

    test_bit_array = &myBitArray;
    actual_bit_array = myBitArray;

    //log(ALWAYS)<<"encoding done, first byte: "<<(unsigned int)(myBitArray.getWrittenBytes()[0])<<endLog();
    //log(ALWAYS)<<"encoding done, second byte: "<<(unsigned int)(myBitArray.getWrittenBytes()[1])<<endLog();
    //log(ALWAYS)<<"encoding done, byte size: "<<myBitArray.getByteSize()<<endLog();
}


std::vector<BitArray> MeshEncoder::getMeshBinaryColor(std::map<int,cv::Mat> &dctTrianglesList, int trianglesNumber)const
{
    std::vector<BitArray>triangleBitArray(trianglesNumber, BitArray(20));

    //write data
    std::map<int,cv::Mat>::iterator it;
    int tri=0;
    //loop over resolution levels
    for(it = dctTrianglesList.begin();it!=dctTrianglesList.end();++it)
    {
        int myRes = it->first;
        cv::Mat patterns = it->second;
        //int sampNum = 3*(myRes-1)*(myRes-2)/2;
        int  sampNum = (myRes+2)*(myRes+1)/2 + (myRes/2+2)*(myRes/2+1);

        //create huffman trees
        HuffTree dcTree = HuffTree();
        HuffTree acTree = HuffTree();

        int patternsNum = patterns.cols;

        int lastDcCoef=0;
        int trailingZeros=0;

        for(int i=0;i<patternsNum;++i)      
        {
            for(int j=0;j<sampNum;++j)
            {
                if(j==0)        //DC coef
                {
                    //log(ALWAYS)<<"writing DC coef "<<int(patterns.at<float>(j,i))<<", previous coef: "<<lastDcCoef<<", (i,j) = ("<<i<<","<<j<<")"<<endLog();
                    addDCCoef(int(patterns.at<float>(j,i))-lastDcCoef, dcTree);    //needs to be per channel?
                    lastDcCoef = int(patterns.at<float>(j,i));
                }
                else                        //AC coef
                {
                    if(int(patterns.at<float>(j,i))==0)
                    {
                        if(j==(sampNum-1))        //end of block, no need to write all the zeros
                        {
                            //log(ALWAYS)<<"writing EOB"<<endLog();
                            acTree.addOccurences((unsigned char)0,1);
                            trailingZeros=0;
                        }
                        else
                        {
                            ++trailingZeros;
                        }
                    }
                    else
                    {
                        //log(ALWAYS)<<"writing AC coef "<<int(patterns.at<float>(j,i))<<" with "<<trailingZeros<<" trailing zeros"<<endLog();
                        addACCoef(int(patterns.at<float>(j,i)), trailingZeros, acTree);
                        trailingZeros=0;
                    }
                }
            }

        }
        //log(ALWAYS)<<"Resolution level:"<<myRes<<endLog();
        //log(ALWAYS)<<"DC Tree:"<<endLog();
        dcTree.computeHuffmanTable();
        //log(ALWAYS)<<"AC Tree:"<<endLog();
        acTree.computeHuffmanTable();

        //recompute everything
        lastDcCoef=0;
        trailingZeros=0;

        for(int i=0;i<patternsNum;++i)  //for each triangle      
        {
            for(int j=0;j<sampNum;++j)
            {
                if(j==0)        //DC coef
                {
                    //log(ALWAYS)<<"writing DC coef "<<int(patterns.at<float>(j,i))<<", previous coef: "<<lastDcCoef<<", (i,j) = ("<<i<<","<<j<<")"<<endLog();
                    writeCodedDCCoef(triangleBitArray[tri], int(patterns.at<float>(j,i))-lastDcCoef, dcTree);    //needs to be per channel?
                    //log(ALWAYS)<<"Byte size: "<<myBitArray.getByteSize()<<endLog();
                    //log(ALWAYS)<<"DC coef written"<<endLog();
                    lastDcCoef = int(patterns.at<float>(j,i));
                }
                else                        //AC coef
                {
                    if(int(patterns.at<float>(j,i))==0)
                    {
                        if(j==(sampNum-1))        //end of block, no need to write all the zeros
                        {
                            triangleBitArray[tri].writeBits(acTree.getCodedByte(0),acTree.getBitNum(0));
                            trailingZeros=0;
                        }
                        else
                        {
                            ++trailingZeros;
                        }
                    }
                    else
                    {
                        //log(ALWAYS)<<"writing AC coef "<<int(patterns.at<float>(j,i))<<" with "<<trailingZeros<<" trailing zeros"<<endLog();
                        writeCodedACCoef(triangleBitArray[tri], int(patterns.at<float>(j,i)), trailingZeros, acTree);
                        trailingZeros=0;
                    }
                }
            }
            triangleBitArray[tri].padLastByte();
            ++tri;
        }
    }
    return triangleBitArray;
}



void MeshEncoder::writeJPEGMeshColor(std::map<int,std::vector<cv::Mat>> &dctTrianglesList, std::string outPath)const
{
    //open file to write in
    //std::ofstream fout("/morpheo-nas2/DataKinovis/CoolCap/TaeKwondo-Thu-Feb-9-12-26-31-2017/kick540/Results/RVDs/Temp/datastreamTest.dat", std::ios::out | std::ios::binary);
    // std::ofstream fout("/morpheo-nas2/DataKinovis/marmando_temp/Results/RVDs/Temp/datastreamTest.dat", std::ios::out | std::ios::binary);
    std::ofstream fout(outPath + "/datastreamTest.dat", std::ios::out | std::ios::binary);


    BitArray myBitArray(5000000);
    
    //Start of Image
    myBitArray.writeBytes(m_SOI);

    //write2BytesNum(myBitArray,1);                 //number of bytes (test here, 1)
    myBitArray.writeSingleByte(0);
    myBitArray.writeSingleByte(1);
    
    //write data
    std::map<int,std::vector<cv::Mat>>::iterator it;

    //loop over resolution levels
    for(it = dctTrianglesList.begin();it!=dctTrianglesList.end();++it)
    {

        int myRes = it->first;
        std::vector<cv::Mat> patterns = it->second;
        int sampNum = (myRes-1)*(myRes-2)/2;

        //create huffman trees
        HuffTree dcYTree = HuffTree();
        HuffTree acYTree = HuffTree();
        HuffTree* myDCTree;
        HuffTree* myACTree;
        HuffTree dcCTree = HuffTree();
        HuffTree acCTree = HuffTree();
        //HuffTree dcCrTree = HuffTree();
        //HuffTree acCrTree = HuffTree();

        int patternsNum = patterns[0].cols;

        int lastDcCoef=0;
        int trailingZeros=0;

        for(int ch=0;ch<3;++ch)
        {
            if(ch==0)       //Y channel
            {
                myDCTree = &dcYTree;
                myACTree = &acYTree;
            }
            else
            {
                myDCTree = &dcCTree;
                myACTree = &acCTree;
            }
            for(int i=0;i<patternsNum;++i)      
            {
                for(int j=0;j<sampNum;++j)
                {

                    if(myBitArray.getByteSize()>1000000)
                    {
                        goto writingFile;
                    }
                    if(j==0)        //DC coef
                    {
                        //log(ALWAYS)<<"writing DC coef "<<int(patterns.at<float>(j,i))<<", previous coef: "<<lastDcCoef<<", (i,j) = ("<<i<<","<<j<<")"<<endLog();
                        addDCCoef(int(patterns[ch].at<float>(j,i))-lastDcCoef, *myDCTree);    //needs to be per channel?
                        lastDcCoef = int(patterns[ch].at<float>(j,i));
                    }
                    else                        //AC coef
                    {
                        if(int(patterns[ch].at<float>(j,i))==0)
                        {
                            if(j==(sampNum-1))        //end of block, no need to write all the zeros
                            {
                                //log(ALWAYS)<<"writing EOB"<<endLog();
                                //myBitArray.writeBytes(m_EOB);
                                (*myACTree).addOccurences((unsigned char)0,1);
                                trailingZeros=0;
                            }
                            else
                            {
                                ++trailingZeros;
                            }
                        }
                        else
                        {
                            //log(ALWAYS)<<"writing AC coef "<<int(patterns.at<float>(j,i))<<" with "<<trailingZeros<<" trailing zeros"<<endLog();
                            addACCoef(int(patterns[ch].at<float>(j,i)), trailingZeros, *myACTree);
                            trailingZeros=0;
                        }
                    }
                }

            }
        }
        //log(ALWAYS)<<"Resolution level:"<<myRes<<endLog();
        //log(ALWAYS)<<"DC Y Tree:"<<endLog();
        dcYTree.computeHuffmanTable();
        //log(ALWAYS)<<"AC Y Tree:"<<endLog();
        acYTree.computeHuffmanTable();


        //recompute everything
        std::vector<int>lastDcCoef3={0,0,0};
        trailingZeros=0;

        for(int i=0;i<patternsNum;++i)      
        {

            for(int ch=0;ch<3;++ch)
            {
                if(ch==0)       //Y channel
                {
                    myDCTree = &dcYTree;
                    myACTree = &acYTree;
                }
                else
                {
                    myDCTree = &dcCTree;
                    myACTree = &acCTree;
                }
                for(int j=0;j<sampNum;++j)
                {

                    if(myBitArray.getByteSize()>1000000)
                    {
                        goto writingFile;
                    }
                    if(j==0)        //DC coef
                    {

                        //log(ALWAYS)<<"writing DC coef "<<int(patterns.at<float>(j,i))<<", previous coef: "<<lastDcCoef<<", (i,j) = ("<<i<<","<<j<<")"<<endLog();
                        writeCodedDCCoef(myBitArray, int(patterns[ch].at<float>(j,i))-lastDcCoef3[ch], *myDCTree);    //needs to be per channel?
                        //log(ALWAYS)<<"Byte size: "<<myBitArray.getByteSize()<<endLog();
                        //log(ALWAYS)<<"DC coef written"<<endLog();
                        lastDcCoef3[ch] = int(patterns[ch].at<float>(j,i));
                    }
                    else                        //AC coef
                    {
                        if(int(patterns[ch].at<float>(j,i))==0)
                        {
                            if(j==(sampNum-1))        //end of block, no need to write all the zeros
                            {
                                //log(ALWAYS)<<"writing EOB"<<endLog();
                                //myBitArray.writeBytes(m_EOB);
                                myBitArray.writeBits((*myACTree).getCodedByte(0),(*myACTree).getBitNum(0));
                                trailingZeros=0;
                            }
                            else
                            {
                                ++trailingZeros;
                            }
                        }
                        else
                        {
                            //log(ALWAYS)<<"writing AC coef "<<int(patterns.at<float>(j,i))<<" with "<<trailingZeros<<" trailing zeros"<<endLog();
                            writeCodedACCoef(myBitArray, int(patterns[ch].at<float>(j,i)), trailingZeros, *myACTree);
                            trailingZeros=0;
                        }
                    }
                }
            }

        }


    }
    writingFile:
    //log(ALWAYS)<<"myBitArray size: "<<myBitArray.getByteSize()<<endLog();
    myBitArray.padLastByte();
    std::vector<unsigned char> byteArray;
    byteArray = myBitArray.getWrittenBytes();
    //log(ALWAYS)<<"myBitArray size: "<<myBitArray.getByteSize()<<endLog();
    //log(ALWAYS)<<"byteArray size: "<<byteArray.size()<<" ("<<sizeof(byteArray[0])<<")"<<endLog();
    fout.write((char*)&byteArray[0],byteArray.size());

    //close file
    fout.close();
}


std::vector<BitArray> MeshEncoder::getMeshBinaryColor(std::map<int,std::vector<cv::Mat>> &dctTrianglesList,int trianglesNumber)const
{
    std::vector<BitArray>triangleBitArray(trianglesNumber, BitArray(20));
    
    //write data
    std::map<int,std::vector<cv::Mat>>::iterator it;
    int tri=0;
    //loop over resolution levels
    for(it = dctTrianglesList.begin();it!=dctTrianglesList.end();++it)
    {

        int myRes = it->first;
        std::vector<cv::Mat> patterns = it->second;
        int sampNum = (myRes-1)*(myRes-2)/2;

        //create huffman trees
        HuffTree dcYTree = HuffTree();
        HuffTree acYTree = HuffTree();
        HuffTree* myDCTree;
        HuffTree* myACTree;
        HuffTree dcCTree = HuffTree();
        HuffTree acCTree = HuffTree();
        //HuffTree dcCrTree = HuffTree();
        //HuffTree acCrTree = HuffTree();

        int patternsNum = patterns[0].cols;

        int lastDcCoef=0;
        int trailingZeros=0;

        for(int ch=0;ch<3;++ch)
        {
            if(ch==0)       //Y channel
            {
                myDCTree = &dcYTree;
                myACTree = &acYTree;
            }
            else
            {
                myDCTree = &dcCTree;
                myACTree = &acCTree;
            }
            for(int i=0;i<patternsNum;++i)      
            {
                for(int j=0;j<sampNum;++j)
                {
                    if(j==0)        //DC coef
                    {
                        //log(ALWAYS)<<"writing DC coef "<<int(patterns.at<float>(j,i))<<", previous coef: "<<lastDcCoef<<", (i,j) = ("<<i<<","<<j<<")"<<endLog();
                        addDCCoef(int(patterns[ch].at<float>(j,i))-lastDcCoef, *myDCTree);    //needs to be per channel?
                        lastDcCoef = int(patterns[ch].at<float>(j,i));
                    }
                    else                        //AC coef
                    {
                        if(int(patterns[ch].at<float>(j,i))==0)
                        {
                            if(j==(sampNum-1))        //end of block, no need to write all the zeros
                            {
                                //log(ALWAYS)<<"writing EOB"<<endLog();
                                //myBitArray.writeBytes(m_EOB);
                                (*myACTree).addOccurences((unsigned char)0,1);
                                trailingZeros=0;
                            }
                            else
                            {
                                ++trailingZeros;
                            }
                        }
                        else
                        {
                            //log(ALWAYS)<<"writing AC coef "<<int(patterns.at<float>(j,i))<<" with "<<trailingZeros<<" trailing zeros"<<endLog();
                            addACCoef(int(patterns[ch].at<float>(j,i)), trailingZeros, *myACTree);
                            trailingZeros=0;
                        }
                    }
                }

            }
        }
        log(ALWAYS)<<"Resolution level:"<<myRes<<endLog();
        log(ALWAYS)<<"DC Y Tree:"<<endLog();
        dcYTree.computeHuffmanTable();
        log(ALWAYS)<<"AC Y Tree:"<<endLog();
        acYTree.computeHuffmanTable();
        log(ALWAYS)<<"DC C Tree:"<<endLog();
        dcCTree.computeHuffmanTable();
        log(ALWAYS)<<"AC C Tree:"<<endLog();
        acCTree.computeHuffmanTable();


        //recompute everything
        std::vector<int>lastDcCoef3={0,0,0};
        trailingZeros=0;

        for(int i=0;i<patternsNum;++i)      //for each triangle
        {

            for(int ch=0;ch<3;++ch)         //for each channel (Y,Cb,Cr)
            {
                if(ch==0)       //Y channel
                {
                    myDCTree = &dcYTree;
                    myACTree = &acYTree;
                }
                else
                {
                    myDCTree = &dcCTree;
                    myACTree = &acCTree;
                }
                for(int j=0;j<sampNum;++j)
                {
                    if(j==0)        //DC coef
                    {

                        //log(ALWAYS)<<"writing DC coef "<<int(patterns.at<float>(j,i))<<", previous coef: "<<lastDcCoef<<", (i,j) = ("<<i<<","<<j<<")"<<endLog();
                        writeCodedDCCoef(triangleBitArray[tri], int(patterns[ch].at<float>(j,i))-lastDcCoef3[ch], *myDCTree);    //needs to be per channel?
                        //log(ALWAYS)<<"Byte size: "<<myBitArray.getByteSize()<<endLog();
                        //log(ALWAYS)<<"DC coef written"<<endLog();
                        lastDcCoef3[ch] = int(patterns[ch].at<float>(j,i));
                    }
                    else                        //AC coef
                    {
                        if(int(patterns[ch].at<float>(j,i))==0)
                        {
                            if(j==(sampNum-1))        //end of block, no need to write all the zeros
                            {
                                //log(ALWAYS)<<"writing EOB"<<endLog();
                                //myBitArray.writeBytes(m_EOB);
                                triangleBitArray[tri].writeBits((*myACTree).getCodedByte(0),(*myACTree).getBitNum(0));
                                trailingZeros=0;
                            }
                            else
                            {
                                ++trailingZeros;
                            }
                        }
                        else
                        {
                            //log(ALWAYS)<<"writing AC coef "<<int(patterns.at<float>(j,i))<<" with "<<trailingZeros<<" trailing zeros"<<endLog();
                            writeCodedACCoef(triangleBitArray[tri], int(patterns[ch].at<float>(j,i)), trailingZeros, *myACTree);
                            trailingZeros=0;
                        }
                    }
                }
            }
            triangleBitArray[tri].padLastByte();
            ++tri;

        }
        //log(ALWAYS)<<"res = "<<myRes<<", tri = "<<tri<<endLog();


    }
    return triangleBitArray;
    //log(ALWAYS)<<"myBitArray size: "<<myBitArray.getByteSize()<<endLog();
    //myBitArray.padLastByte();
    //std::vector<char> byteArray;
    //byteArray = myBitArray.getWrittenBytes();
    //log(ALWAYS)<<"myBitArray size: "<<myBitArray.getByteSize()<<endLog();
    //log(ALWAYS)<<"byteArray size: "<<byteArray.size()<<" ("<<sizeof(byteArray[0])<<")"<<endLog();
}



void MeshEncoder::write2BytesNum(std::vector<char> &byteArray, unsigned short myNum)const
{
    unsigned short c1 = (unsigned short)(myNum/256);
    unsigned short c2 = myNum - c1;
    unsigned char my2BShort[2] = {c1,c2};
    byteArray.insert(byteArray.end(),&my2BShort[0],&my2BShort[2]);
    //myOS.write((char*)my2BShort,2);
    //myOS.write((char*)c2,1);
}

void MeshEncoder::addDCCoef(int myCoef, HuffTree &myHT)const
{
    //WARNING!!!!!
    //No check that number can be written in less than... 255 bits?
    //Hmm, ok, probably no need...
    //Though the 1st 4 bits are supposed to stay 0 according to the specs. Anyway. Should never reach 16 extrabits either
    
    unsigned int absCoef = std::abs(myCoef);
    
    //Just reimplement JPEG table

    int extraBits=0;
    while(absCoef>=pow(2,extraBits)) //this loop gives us the 'size', i.e. number of extra bits necessary to encode the DC value
    {
        ++extraBits;
    }
    myHT.addOccurences((unsigned char)extraBits,1);

}


void MeshEncoder::writeCodedDCCoef(BitArray &myBitArray, int myCoef, HuffTree &myHT)const
{
    //WARNING!!!!!
    //No check that number can be written in less than... 255 bits?
    //Hmm, ok, probably no need...
    //Though the 1st 4 bits are supposed to stay 0 according to the specs. Anyway. Should never reach 16 extrabits either
    unsigned int absCoef = std::abs(myCoef);
    
    //Just reimplement JPEG table

    int extraBits=0;
    while(absCoef>=pow(2,extraBits)) //this loop gives us the 'size', i.e. number of extra bits necessary to encode the DC value
    {
        ++extraBits;
    }

    //write size on 1 byte (coded with Huffman table)
    myBitArray.writeBits(myHT.getCodedByte(extraBits),myHT.getBitNum(extraBits));

    if(myCoef>0)        //if positive number, just write binary value
    {
        myBitArray.writeBits(absCoef,extraBits);
    }
    else if (myCoef<0)                //negative number: write complement
    {
        myBitArray.writeBits((unsigned int)(std::pow(2,extraBits)-1-absCoef),extraBits);
    }
    //3rd case: myCoef = 0. no additional bits to write

}

void MeshEncoder::addACCoef(int myCoef, int trailingZeros, HuffTree &myHT)const
{
    if(trailingZeros>15)
    {
        //myBitArray.writeSingleByte(240);     //240 = 1111 0000 = (15,0)
        myHT.addOccurences((unsigned char)240,1);
        addACCoef(myCoef, trailingZeros-16, myHT);
    }
    else    //similar to DC coefficient, but the first 4 bits (runlength) before size are used to encode the number of trailing zeros
    {
        unsigned char runlength = trailingZeros;
        runlength = runlength << 4; //left shift to 1st four bits
        //Then, same thing as DC coefficient
        unsigned int absCoef = std::abs(myCoef);
        int extraBits=0;
        while(absCoef>=pow(2,extraBits)) //this loop gives us the 'size', i.e. number of extra bits necessary to encode the DC value
        {
            ++extraBits;
        }
        //This part differs slightly from DCCoef: write size in the same byte as runlength
        runlength+=extraBits;
        myHT.addOccurences(runlength,1);
    }
}


void MeshEncoder::writeCodedACCoef(BitArray &myBitArray, int myCoef, int trailingZeros, HuffTree &myHT)const
{
    if(trailingZeros>15)
    {
        myBitArray.writeBits(myHT.getCodedByte(240),myHT.getBitNum(240));
        writeCodedACCoef(myBitArray,myCoef, trailingZeros-16, myHT);
    }
    else    //similar to DC coefficient, but the first 4 bits (runlength) before size are used to encode the number of trailing zeros
    {
        unsigned char runlength = trailingZeros;
        runlength = runlength << 4; //left shift to 1st four bits
        //Then, same thing as DC coefficient
        unsigned int absCoef = std::abs(myCoef);
        int extraBits=0;
        while(absCoef>=pow(2,extraBits)) //this loop gives us the 'size', i.e. number of extra bits necessary to encode the DC value
        {
            ++extraBits;
        }
        //This part differs slightly from DCCoef: write size in the same byte as runlength
        runlength+=extraBits;
        myBitArray.writeBits(myHT.getCodedByte(runlength),myHT.getBitNum(runlength));
        
        //writing extra bits: exactly the same as DC
        if(myCoef>0)        //if positive number, just write binary value
        {
            myBitArray.writeBits(absCoef,extraBits);
        }
        else                //negative number: write complement
        {
            myBitArray.writeBits((unsigned int)(std::pow(2,extraBits)-1-absCoef),extraBits);
        }
        //No third case here. 'myCoef' is necessarily not zero

    }
}

std::map<int,cv::Mat> MeshEncoder::decodeCompressedData(std::map<int,cv::Mat> &resPCAEigenVectors, std::string filePath, std::string fileName)
{
	//decode header.
	//Get Huffman table(s) and quantization matrix(ces).
	//TODO

	//read bit stream and decode it using Huffman tables, quantization matrix, and predefined codes

	//BitArray bitStream = *test_bit_array;
	
    // BitArray bitStream = actual_bit_array;
    BitArray bitStream = BitArray(filePath+fileName);

	log(ALWAYS)<<"Decoding data..."<<endLog();

	bitStream.initializeReading();
    
 //    log(ALWAYS)<<"actual_bit_array, byte size: "<<actual_bit_array.getByteSize()<<endLog();
	// log(ALWAYS)<<"actual_bit_array, first byte: "<<(unsigned int)(actual_bit_array.getWrittenBytes()[0])<<endLog();
 //    log(ALWAYS)<<"actual_bit_array, second byte: "<<(unsigned int)(actual_bit_array.getWrittenBytes()[1])<<endLog();
    

    log(ALWAYS)<<"bitStream, byte size: "<<bitStream.getByteSize()<<endLog();
    log(ALWAYS)<<"bitStream, first byte: "<<(unsigned int)(bitStream.getWrittenBytes()[0])<<endLog();
    log(ALWAYS)<<"bitStream, second byte: "<<(unsigned int)(bitStream.getWrittenBytes()[1])<<endLog();
    

	//reading Start of Image marker
	unsigned char b = bitStream.readOneByte();
	if(b!=255)
	{
		log(ALWAYS)<<"Error SOI: "<<(unsigned int)(b)<<endLog();
		log(ALWAYS)<<"written bytes(0): "<<(unsigned int)bitStream.getWrittenBytes()[0]<<endLog();
		log(ALWAYS)<<"byte size: "<<bitStream.getByteSize()<<endLog();
		log(ALWAYS)<<"Remaining bytes: "<<(unsigned int)bitStream.getRemainingBytesNumber()<<endLog();
	}
	b = bitStream.readOneByte();
	if(b!=216)
	{
		log(ALWAYS)<<"Error SOI(2)"<<(unsigned int)(b)<<endLog();
		log(ALWAYS)<<"Remaining bytes: "<<(unsigned int)bitStream.getRemainingBytesNumber()<<endLog();
	}


	short curRes=0;
	std::map<int,cv::Mat> resPCAVectors;
	//int samplesNumber;
	//int chromaSamplesNumber;
	int coefNumber;
	int tri=0;
	int previousDCValue=0;
	int readACCoefsNumber=0;
	HuffTree curDCTree;
	HuffTree curACTree;
	std::vector<std::vector<int> > pcaColorsV;
	std::vector<float> myQT;

	//for each byte
	while(bitStream.getRemainingBytesNumber()>0)
	{
		int savedBitOffset = 8-bitStream.getReadingBitOffset();
		if(savedBitOffset==8)
		{
			savedBitOffset=0;
		}
		bitStream.readBitsChar(savedBitOffset);	//go to next complete byte to check for marker
																	//just added padding before writing markers makes debugging easier.
																	//bad way of doing this (reading and going back every time...) Need a better solution (TODO)

		//check resolution change somehow
		unsigned char b = bitStream.readOneByte();
		if(b==255)		//FF. Could be a JPEG code. Need to make sure this doesn't happen in the encoded data (by adding extra byte when FF appears (?))
		{				//For now, we assume there is no ambiguity
			b = bitStream.readOneByte();	//read next byte. Determines what code this is
			if(b==228)		//TODO
			{				//We have a resolution change marker
				b = bitStream.readOneByte();					//add second marker for now, until we add stuff bytes
				unsigned char b2 = bitStream.readOneByte();
				if((b==255)&&(b2==228))
				{
					//First, add current working data to final result
					if(curRes>0)
					{
						cv::Mat tempMat(pcaColorsV.size(),coefNumber,CV_32FC1,0.0f);
						for(int i=0; i<tempMat.rows;++i)
						{
							for(int j=0;j<tempMat.cols;++j)
							{
								tempMat.at<float>(i,j)=float(pcaColorsV[i][j])*float(myQT[j]);
							}
						}
						resPCAVectors[curRes] = tempMat;
					}


					curRes = short(bitStream.readOneByte());	//new resolution is encoded on one byte

					// if(curRes==16)
					// {
					// 	log(ALWAYS)<<"switching to res 16: remaining bytes: "<<bitStream.getRemainingBytesNumber()<<", bit offset: "<<bitStream.getReadingBitOffset()<<endLog();
					// 	int tempInd = bitStream.getReadingByteIndex();
					// 	log(ALWAYS)<<"reading byte index = "<<tempInd<<endLog();
					// 	log(ALWAYS)<<"("<<int(bitStream.getWrittenBytes()[tempInd-2])<<","<<int(bitStream.getWrittenBytes()[tempInd-1])<<","<<int(bitStream.getWrittenBytes()[tempInd])<<","
					// 		<<int(bitStream.getWrittenBytes()[tempInd+1])<<","<<int(bitStream.getWrittenBytes()[tempInd+2])<<")"<<endLog();
					// }

					//samplesNumber = (curRes+2)*(curRes+1)/2;
					//chromaSamplesNumber = (curRes/2+2)*(curRes/2+1)/2;
					//coefNumber = samplesNumber+2*chromaSamplesNumber;
					cv::Mat pcaColors;			//TODO: get number of triangles and create matrix with the right dimensions (and type)
					pcaColorsV.clear();
					tri = 0;
					previousDCValue=0;

					//log(ALWAYS)<<"Resolution change: new res="<<curRes<<endLog();
					//log(ALWAYS)<<"Remaining bytes: "<<bitStream.getRemainingBytesNumber()<<endLog();
					//Read Huffman trees (for now)
					curDCTree = readHuffTree(bitStream);
					curACTree = readHuffTree(bitStream);

					
					int myQTNumber = readQuantizationMatrix(bitStream,myQT);
					coefNumber = myQT.size();

                    int colorSpaceSize = (curRes+2)*(curRes+1)/2 + (curRes/2+2)*(curRes/2+1)-9;
                    cv::Mat pcaEigenVectors = cv::Mat(coefNumber+1,colorSpaceSize,CV_16SC1);        //+1 for mean vector (works like an additional component)
                    int pcaVecNumber = readPCAEigenVectors(bitStream,pcaEigenVectors);

                    resPCAEigenVectors[curRes] = pcaEigenVectors;
					//log(ALWAYS)<<"Huffman trees read"<<endLog();
					//log(ALWAYS)<<"Remaining bytes: "<<bitStream.getRemainingBytesNumber()<<endLog();

					//reading Start of Scan
					b = bitStream.readOneByte();
					if(b!=255)
					{
						log(ALWAYS)<<"Error SOS: "<<(unsigned int)(b)<<endLog();
					}
					b = bitStream.readOneByte();
					if(b!=218)
					{
						log(ALWAYS)<<"Error SOS(2): "<<(unsigned int)(b)<<endLog();
					}
				}
				else
				{
					bitStream.moveBackOneByte();
					bitStream.moveBackOneByte();
					bitStream.moveBackOneByte();
					bitStream.moveBackOneByte();
					bitStream.moveBackNBits(savedBitOffset);
					log(ALWAYS)<<"False alert(2)"<<endLog();
				}
				//curDCTree = &;		//TODO
				//curACTree = &;		//TODO
			}
			else		//false alert. Assume this was an artifact of the huffman encoding. Will need something more robust in the future. And check for other possible markers too?
			{			//Maybe we'll have to signal huffman table changes and quantization matrix changes here too
				bitStream.moveBackOneByte();
				bitStream.moveBackOneByte();
				bitStream.moveBackNBits(savedBitOffset);
				//log(ALWAYS)<<"False alert"<<endLog();
			}
		}
		else
		{
			bitStream.moveBackOneByte();	//no JPEG marker. Move back cursor so the last byte can be read again
			bitStream.moveBackNBits(savedBitOffset);
			//log(ALWAYS)<<"False alert (1)"<<endLog();
		}

		//process triangle. Keep the loop going until broken by another resolution change

		std::vector<int> trianglePCAComp;
		trianglePCAComp.reserve(coefNumber);
		//log(ALWAYS)<<"Reading DC Coef..."<<endLog();
		int myDCCoef = readDCCoef(bitStream,curDCTree,previousDCValue);
		
		previousDCValue = myDCCoef;
		trianglePCAComp.push_back(myDCCoef);
		//return resPCAVectors;

		//read AC coefs
		//log(ALWAYS)<<"Reading AC Coefs..."<<endLog();
		readACCoefsNumber=0;
		while(readACCoefsNumber<(coefNumber-1))		//keep going until we reach the expected number of AC coefficients, or a End of Block statement.
		{
			
			int trailingZeros;
			int myACCoef = readACCoef(bitStream,curACTree,trailingZeros);
			
			if((myACCoef==0)&&(trailingZeros==0))		//End of Block
			{
				trianglePCAComp.insert(trianglePCAComp.end(),coefNumber-1-readACCoefsNumber,0);
				readACCoefsNumber = coefNumber-1;
			}
			else
			{
				trianglePCAComp.insert(trianglePCAComp.end(),trailingZeros,0);	//insert zeros
				trianglePCAComp.push_back(myACCoef);
				readACCoefsNumber+=trailingZeros+1;
			}
		}
		//log(ALWAYS)<<"AC Coefs read..."<<endLog();

		//add vector to matrix
		pcaColorsV.push_back(trianglePCAComp);
		++tri;

	}

	// log(ALWAYS)<<"res = "<<curRes<<"decoded quantization matrix: "<<endLog();
	// log(ALWAYS)<<"[";
	// 	for(int vInd=0;vInd<myQT.size();++vInd)
	// 	{
	// 		log(ALWAYS)<<myQT[vInd]<<",";
	// 	}
	// log(ALWAYS)<<"]"<<endLog();
	//Add final resolution
	cv::Mat tempMat(pcaColorsV.size(),coefNumber,CV_32FC1,0.0f);
	for(int i=0; i<tempMat.rows;++i)
	{
		for(int j=0;j<tempMat.cols;++j)
		{
			//tempMat.at<float>(i,j)=float(pcaColorsV[i][j]);
			tempMat.at<float>(i,j)=float(pcaColorsV[i][j])*float(myQT[j]);
		}
	}
	resPCAVectors[curRes] = tempMat;
	//log(ALWAYS)<<"Adding matrix of size ("<<tempMat.rows<<","<<tempMat.cols<<") to res "<<curRes<<endLog();

	return resPCAVectors;
}

//This reads the first n bits of the bit stream until it matches an encoded value. returns it
unsigned char MeshEncoder::readCodedByte(BitArray &bitStream, HuffTree &myTree)
{
	unsigned int codedValue=0;
	unsigned int bitNum=0;
	while(true)
	{
		codedValue += bitStream.readBitsInt(1);
		++bitNum;
		//log(ALWAYS)<<"codedValue = "<<codedValue<<", bitNum = "<<bitNum<<endLog();
		for(int i=0;i<256;++i)
		{
			//log(ALWAYS)<<"i = "<<int(i)<<endLog();
			//log(ALWAYS)<<"bitsnum size: "<<myTree.getBitsNumSize()<<endLog();
			//log(ALWAYS)<<"bitnum[i] = "<<myTree.getBitNum(i)<<endLog();
			//log(ALWAYS)<<"codedByte[i] = "<<myTree.getCodedByte(i)<<endLog();
			if(myTree.getBitNum(i)==bitNum && myTree.getCodedByte(i)==codedValue)
			{
				return i;
			}
			if(i==255)	//we need this trick to get out of the loop with chars
			{
				break;
			}
		}
		//log(ALWAYS)<<"no match yet. "<<endLog();
		//no match found yet. Read next bit
		codedValue=codedValue<<1;
		//log(ALWAYS)<<"codedValue updated: "<<codedValue<<endLog();
		if(bitNum>24)
		{
			log(ALWAYS)<<"ABOOOOOOOOOOOOOOOOORT"<<endLog();
			log(ALWAYS)<<bitStream.getRemainingBytesNumber()<<" remaining bytes"<<endLog();
			return 0;
		}
	}
}

//called when we expect a DC coefficient. Reads first the huffman-encoded number of bits, then reads those and deduce value
int MeshEncoder::readDCCoef(BitArray &bitStream, HuffTree &dcTree, int previousDCValue)
{
	unsigned char bitNum = readCodedByte(bitStream, dcTree);
	//log(ALWAYS)<<"DC coef. bitNum = "<<bitNum<<endLog();

	int rawCoef = bitStream.readBitsInt(bitNum);
	//log(ALWAYS)<<"DC Coef. rawCoef = "<<rawCoef<<endLog();
	int encodedValue;
	int pivotValue = std::pow(2,bitNum-1);
	//log(ALWAYS)<<"DC Coef. pivotValue = "<<pivotValue<<endLog();
	
	if(rawCoef>=pivotValue)		//first bit is 1, positive value
	{
		encodedValue = rawCoef;
	}
	else						//negative value
	{
		encodedValue = rawCoef - 2*pivotValue + 1;
	}
	//log(ALWAYS)<<"DC Coef. encodedValue= "<<encodedValue<<endLog();
	return (encodedValue + previousDCValue);
}


int MeshEncoder::readACCoef(BitArray &bitStream, HuffTree &acTree, int &out_trailingZeros)
{
	unsigned char bitNum = readCodedByte(bitStream, acTree);
	//log(ALWAYS)<<"AC Coef. bitNum = "<<bitNum<<endLog();
	//special case: End of Block (byte with value of zero)
	if(bitNum==0)
	{
		out_trailingZeros=0;
		return 0;
	}

	out_trailingZeros = bitNum/16;		//first 4 bits encode the number of zeros preceding the encoded non zero value
	//log(ALWAYS)<<"AC Coef. out_trailingZeros = "<<out_trailingZeros<<endLog();
	bitNum = bitNum - out_trailingZeros*16;		//last 4 bits encode the number of bits used to encode the actual value
	//log(ALWAYS)<<"AC Coef. bitNum = "<<bitNum<<endLog();
	
	if(bitNum==0)	//(15,0) case for encoding 16 consecutive zeros before the end of block. Theoretically (and empirically) it should work with the normal handling.
	{				//But this seems cleaner somehow.
		return 0;
	}
	int rawCoef = bitStream.readBitsInt(bitNum);
	int encodedValue;
	int pivotValue = std::pow(2,bitNum-1);
	//log(ALWAYS)<<"AC Coef. rawCoef = "<<rawCoef<<endLog();

	if(rawCoef>=pivotValue)		//first bit is 1, positive value
	{
		encodedValue = rawCoef;
	}
	else						//negative value
	{
		encodedValue = rawCoef - 2*pivotValue + 1;
	}
	//log(ALWAYS)<<"AC Coef. encodedValue = "<<encodedValue<<endLog();
	return encodedValue;
}

//std::vector<unsigned char> MeshEncoder::readMarker(BitArray &bitStream)
//{
//
//}


HuffTree MeshEncoder::readHuffTree(BitArray &bitStream)
{
	unsigned char b;
	
	b = bitStream.readOneByte();
	if(b!=255)
	{
		log(ALWAYS)<<"DHT MARKER EXPECTED!!!"<<endLog();
		return HuffTree();
	}
	b = bitStream.readOneByte();
	if(b!=196)
	{
		log(ALWAYS)<<"DHT MARKER EXPECTED(2)!!!"<<endLog();
		return HuffTree();
	}

	//read size (2 bytes)
	int dht_size = bitStream.readBitsInt(16);
	//log(ALWAYS)<<"DHT size: "<<dht_size<<endLog();


	std::vector<int> codesNumber (24);	//16 originally, can't figure out why and how to alter Huffman tables.
										//This trick will do for now.
	std::vector<int> bitsNum(256,0);
	int totalCodes = 0;
	//log(ALWAYS)<<"codesNumber: (";
	for(int i=0;i<24;++i)
	{
		codesNumber[i] = (unsigned int)(bitStream.readOneByte());
		totalCodes+= codesNumber[i];
		//log(ALWAYS)<<codesNumber[i]<<",";
	}
	//log(ALWAYS)<<")"<<endLog();
	int curLength = 1;
	

	for(int i=0;i<totalCodes;++i)
	{
		int word = bitStream.readOneByte();
		while(codesNumber[curLength-1]==0)	//get next valid code length
		{
			++curLength;
		}
		bitsNum[word] = curLength;
		--codesNumber[curLength-1];
	}

	HuffTree readTree = HuffTree(bitsNum);
	readTree.computeCodedBytes();
	return readTree;
}

void MeshEncoder::writeHuffTree(BitArray &myBitArray, HuffTree &myHT)const
{
	std::vector<int> codesNumber (24,0);	//16 originally, can't figure out why and how to alter Huffman tables.
										//This trick will do for now.
	int totalCodes = 0;
	for(int i=0;i<256;++i)
	{
		int bNum = myHT.getBitNum(i);
		if (bNum>0)
		{
			codesNumber[bNum-1]+=1;
			++totalCodes;
		}
	}
	//log(ALWAYS)<<"Writing hufftree."<<endLog();

	// log(ALWAYS)<<"codesNumber: (";
	// for(int i=0;i<24;++i)
	// {
	// 	log(ALWAYS)<<codesNumber[i]<<",";
	// }
	// log(ALWAYS)<<")"<<endLog();
	

	//log(ALWAYS)<<"totalCodes = "<<totalCodes<<endLog();
	myBitArray.writeBytes(m_DHT);
	myBitArray.writeBits(totalCodes+24+2,16);	//2 for size bytes (these very ones), 24 for code lengths, varying size for actual codes

	for(int i=0;i<24;++i)
	{
		myBitArray.writeSingleByte(codesNumber[i]);
	}

	int bLength =1;
	unsigned char curCode=0;

	while(bLength<codesNumber.size()+1)
	{
		if(codesNumber[bLength-1]==0)
		{
			++bLength;
			curCode=0;	
		}
		else	//look for code with the right length
		{
			if(myHT.getBitNum(curCode)==bLength)
			{
				myBitArray.writeSingleByte(curCode);
				--codesNumber[bLength-1];
			}
			++curCode;
		}
	}
}

void MeshEncoder::encodeQuantizationMatrix(BitArray &bitStream, std::vector<float> quantMat, int qtNumber = 0)
{
	//write marker
	bitStream.writeBytes(m_DQT);
	//write segment size (two bytes)
	int matLength = quantMat.size();
	int segmentLength = matLength+2+1;
	bitStream.writeBits(segmentLength,16);
	//write table info (QT Number, QT Precision)
	unsigned char qtInfo = qtNumber<<4;
	unsigned char qtPrecision = 1;		//what is this?
	qtInfo+=qtPrecision;
	bitStream.writeSingleByte(qtInfo);
	//write matrix
	for(int i=0;i<matLength;++i)
	{
		unsigned char coef = (unsigned char)(quantMat[i]);
		bitStream.writeSingleByte(coef);
	}
}

void MeshEncoder::encodeEigenVectors(BitArray &bitStream, cv::Mat &eigenVectors, int vectorsNumber, int vNumber = 0)
{
    int bitPrecision=16;
    int segmentLength = int(ceil(eigenVectors.cols * vectorsNumber * (float(bitPrecision)/8) + 2 + 1)+0.1);  //+ 2 for size bytes, +1 for matrix info
    
    log(ALWAYS)<<"Encoding eigen vectors:"<<endLog();
    log(ALWAYS)<<"columns: "<<eigenVectors.cols<<", rows: "<<eigenVectors.rows<<", components kept: "<<vectorsNumber<<", byte size: "<<float(bitPrecision)/8.0<<endLog();
    log(ALWAYS)<<"segmentLength = "<<segmentLength<<endLog();
    //write marker
    bitStream.writeBytes(m_APP5);
    //write size
    bitStream.writeBits(segmentLength,16);

    //write table info (Decomposition Number, Element Precision) inspired from quantization matrix
    unsigned char vInfo = vNumber<<4;
    unsigned char vPrecision = bitPrecision%16;      //used for bitNum per element. 0 means 16
    vInfo+=vPrecision;
    bitStream.writeSingleByte(vInfo);

    for(int i=0;i<vectorsNumber;++i)
    {
        for(int j=0;j<eigenVectors.cols;++j)
        {
            //write value
            int myValue = int(eigenVectors.at<float>(i,j));
            //first, write sign of 1st bit
            if(myValue<0)
            {
                bitStream.writeBits(1,1);
                myValue +=std::pow(2,(bitPrecision-1));         //map negative range the standard way for signed integers
            }
            else
            {
                bitStream.writeBits(0,1);
            }
            bitStream.writeBits(myValue,bitPrecision-1);
        }
    }
    bitStream.padLastByte();

}

int MeshEncoder::readQuantizationMatrix(BitArray &bitStream, std::vector<float> &out_QT)
{
	unsigned char b;
	b = bitStream.readOneByte();
	if(b!=255)
	{
		log(ALWAYS)<<"DQT MARKER EXPECTED!!!"<<endLog();
		return -1;
	}
	b = bitStream.readOneByte();
	if(b!=219)
	{
		log(ALWAYS)<<"DQT MARKER EXPECTED(2)!!!"<<endLog();
		return -1;
	}

	//read size (2 bytes)
	int dqt_size = bitStream.readBitsInt(16);
	//log(ALWAYS)<<"DQT size: "<<dqt_size<<endLog();
	//read table info
	int qtInfo = bitStream.readOneByte();
	int qtNumber = qtInfo / 16;
	int qtPrecision = qtInfo % 16;

	//read table
	int tableLength = dqt_size-3;
	out_QT.clear();
	out_QT.reserve(tableLength);
	for(int i=0;i<tableLength;++i)
	{
		float coef = float(bitStream.readOneByte());
		out_QT.push_back(coef);
	}
	return qtNumber;  
}

int MeshEncoder::readPCAEigenVectors(BitArray &bitStream, cv::Mat &eigenVectors)
{
    unsigned char b;
    b = bitStream.readOneByte();
    if(b!=255)
    {
        log(ALWAYS)<<"APP5 MARKER EXPECTED!!!"<<endLog();
        return -1;
    }
    b = bitStream.readOneByte();
    if(b!=229)
    {
        log(ALWAYS)<<"APP5 MARKER EXPECTED(2)!!!"<<endLog();
        return -1;
    }

    //read size (2 bytes)
    int eigenvec_size = bitStream.readBitsInt(16);
    
    //read table info
    int vInfo = bitStream.readOneByte();
    int vNumber = vInfo / 16;
    int vPrecision = vInfo % 16;
    if(vPrecision==0)
    {
        vPrecision=16;
    }
    log(ALWAYS)<<"Read precision: "<<vPrecision<<endLog();
    int readValues=0;
    for(int i=0;i<eigenVectors.rows;++i)
    {
        for(int j=0;j<eigenVectors.cols;++j)
        {
            unsigned char sign = bitStream.readBitsChar(1);
            int test = bitStream.readBitsInt(vPrecision-1);
            if(sign==1)
            {
                test -= std::pow(2,vPrecision-1);
            }
            eigenVectors.at<short>(i,j) = test;
            ++readValues;
        }
    }
    //log(ALWAYS)<<"eigenVectors mat size: ("<<eigenVectors.rows<<","<<eigenVectors.cols<<"), read values: "<<readValues<<endLog();
    //bitStream.readBitsInt((eigenvec_size-3)*8);
    bitStream.readEndOfByte();
    return vNumber;
}