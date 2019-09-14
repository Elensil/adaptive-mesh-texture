
#ifndef BIT_ARRAY
#define BIT_ARRAY



#include "matrix.h"
#include "optionmanager.h"


#include "stdlib.h"
#include <list>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include <map>
#include <iostream>


class BitArray{

private:
    std::vector<unsigned char> writtenBytes;
    unsigned char incompleteByte;
    int waitingBits;
    //number of bytes already fully read, or index of current byte
    int readingBitOffset;
    // number of bits already read in current byte, or index of starting bit in current byte
    int readingByteIndex;


public:
    BitArray(){
    	incompleteByte=0;
    	waitingBits=0;
    	writtenBytes.clear();
        readingBitOffset=0;
        readingByteIndex=0;
    };

    BitArray(int length){
    	incompleteByte=0;
    	waitingBits=0;
    	writtenBytes.clear();
    	writtenBytes.reserve(length);
        readingBitOffset=0;
        readingByteIndex=0;
    };

    BitArray(std::string fileName){

        log(ALWAYS)<<"Load binary file "<<fileName<<endLog();
        incompleteByte=0;
        waitingBits=0;
        readingBitOffset=0;
        readingByteIndex=0;
        writtenBytes.clear();

        // std::basic_ifstream<unsigned char> fin(fileName, std::ios::in | std::ios::binary);
        // fin.unsetf(std::ios::skipws);
        // std::streampos fileSize;
        // fin.seekg(0, std::ios::end);
        // fileSize = fin.tellg();
        // fin.seekg(0, std::ios::beg);
        // writtenBytes.reserve(fileSize);
        // writtenBytes.insert(writtenBytes.begin(),std::istream_iterator<unsigned char>(fin),std::istream_iterator<unsigned char>());
        // //std::copy(std::istream_iterator<unsigned char>(fin),std::istream_iterator<unsigned char>(),std::back_inserter(writtenBytes));
        
        // std::ifstream in(fileName);
        // std::string contents((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        // writtenBytes = contents.c_str();

        std::ifstream in(fileName); //open file
        in >> std::noskipws;  //we don't want to skip spaces        
        //initialize a vector with a pair of istream_iterators
        std::vector<unsigned char> v((std::istream_iterator<unsigned char>(in)), 
                                (std::istream_iterator<unsigned char>()));

        log(ALWAYS)<<"v size = "<<v.size()<<endLog();
        writtenBytes = v;
        log(ALWAYS)<<"writtenBytes size = "<<writtenBytes.size()<<endLog();
        
    }

    void writeSingleBit(bool myBit);

    void writeSingleByte(unsigned char myByte);

    void padLastByte();

    void writeBits(unsigned char myByte, int bitNum);	//write a value that can be stored in a single byte. bitNum is <= 8

    void writeBits(unsigned int myInt, int bitNum);				//write any (positive) int. bitNum can be bigger than 8

    inline void writeBits(int myInt, int bitNum){writeBits((unsigned int)myInt, bitNum);}

    void writeBytes(const unsigned char myBytes[]);

    void write2Bytes(unsigned char myBytes[]);

    void writeBytes(const std::vector<unsigned char> &myBytes);

    inline int getWaitingBits()const{return waitingBits;}

    inline unsigned char getIncompleteByte()const{return incompleteByte;}

    inline std::vector<unsigned char> getWrittenBytes()const{return writtenBytes;}

    inline int getReadingBitOffset()const{return readingBitOffset;}

    inline int getReadingByteIndex()const{return readingByteIndex;}

    inline int getByteSize()const{return writtenBytes.size();}

    inline int getBitSize()const{return 8*writtenBytes.size()+waitingBits;}

    //return the number of bits left to be read
    inline int getRemainingBitsNumber()const{return 8*(writtenBytes.size()-readingByteIndex)+waitingBits-readingBitOffset;}

    inline int getRemainingBytesNumber()const{return int(getRemainingBitsNumber()/8);}

    unsigned char readOneByte();

    //returns value on a single byte. bitNum must be <=8
    unsigned char readBitsChar(int bitNum);

    //For bitNum > 8, return value in a int
    unsigned int readBitsInt(int bitNum);

    void moveBackOneByte();

    void moveBackNBits(int bitNum);

    unsigned char readEndOfByte();

    void initializeReading();


};

class HuffNode{

private:
    HuffNode* left;
    HuffNode* right;
    float proba;
    bool isLeaf;
    int* bitsNum;

public:
    //HuffNode(){
    //    left = NULL;
    //    right = NULL;
    //    proba = 0;
    //}

    HuffNode(int* myBitsNum, float myProba){
        left = NULL;
        right = NULL;
        proba = myProba;
        isLeaf = true;
        bitsNum = myBitsNum;
    }

    HuffNode(HuffNode* myLeft, HuffNode* myRight){
        left = myLeft;
        right = myRight;
        bitsNum = NULL;
        proba = left->getProba() + right->getProba();
        isLeaf = false;
    }

    inline float getProba()const{return this->proba;}

    void increment(int depth);

};

class HuffTree{

private:
    std::vector<int> occurences;
    std::vector<int> bitsNum;
    std::vector<int> codedBytes;    //can actually be bigger than a byte (255).

public:
    HuffTree(){
        occurences.assign(256,0);
        bitsNum.assign(256,0);
        codedBytes.assign(256,0);
    }

    HuffTree(std::vector<int> in_bitsNum){
        occurences.assign(256,0);
        bitsNum = in_bitsNum;
        codedBytes.assign(256,0);
    }

    void computeHuffmanTable();

    void computeCodedBytes();

    void addOccurences(unsigned char myCode, int occurencesNum);

    inline int getBitNum(unsigned char i)const{return bitsNum[i];}

    inline int getBitNum(int i)const{return bitsNum[i];}

    inline int getCodedByte(unsigned char i)const{return codedBytes[i];}

    inline int getCodedByte(int i)const{return codedBytes[i];}

    inline int getBitsNumSize()const{return bitsNum.size();}



};

#endif	// BIT_ARRAY