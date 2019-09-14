#include "../Include/bit_array.h"


void BitArray::writeSingleBit(bool myBit)
{
	incompleteByte = incompleteByte<<1;
	if(myBit)
	{
		++incompleteByte;
	}
	++waitingBits;

	if (waitingBits==8)
	{
		writtenBytes.push_back(incompleteByte);
		incompleteByte=0;
		waitingBits=0;
	}
}

void BitArray::writeSingleByte(unsigned char myByte)
{
	

	if(waitingBits==0)			//easy case
	{
		writtenBytes.push_back(myByte);
	}
	else
	{
		unsigned char divider = std::pow(2,waitingBits);
		unsigned char quotient = myByte / divider;
		unsigned char rmdr = myByte - quotient * divider;
		incompleteByte = incompleteByte << (8-waitingBits);
		incompleteByte+=quotient;
		writtenBytes.push_back(incompleteByte);
		incompleteByte = rmdr;
	}
}

void BitArray::writeBits(unsigned char myByte, int bitNum)
{
	//log(ALWAYS)<<"current state: incompleteByte="<<int(getIncompleteByte())<<", waitingBits="<<waitingBits<<endLog();
	if(waitingBits+bitNum<8)	//byte still not complete
	{
		//log(ALWAYS)<<"not completing byte, just add bits to it"<<endLog();
		incompleteByte=incompleteByte<<bitNum;
		incompleteByte+=myByte;
		waitingBits+=bitNum;
	}
	else if(waitingBits+bitNum==8)	//byte completed, no remainder
	{
		//log(ALWAYS)<<"just finishing byte"<<endLog();
		incompleteByte=incompleteByte<<bitNum;
		incompleteByte+=myByte;
		writtenBytes.push_back(incompleteByte);
		incompleteByte=0;
		waitingBits=0;
	}
	else		//writing more than current byte
	{
		//log(ALWAYS)<<"finishing byte and more"<<endLog();
		incompleteByte=incompleteByte<<(8-waitingBits);
		unsigned char divider = std::pow(2,waitingBits+bitNum-8);
		unsigned char quotient = myByte / divider;
		unsigned char rmdr = myByte - quotient * divider;
		incompleteByte+=quotient;
		writtenBytes.push_back(incompleteByte);
		incompleteByte = rmdr;
		waitingBits += (bitNum-8);
	}
	//log(ALWAYS)<<"current state: incompleteByte="<<int(getIncompleteByte())<<", waitingBits="<<waitingBits<<endLog();
}

void BitArray::writeBits(unsigned int myInt, int bitNum)
{
	if(bitNum>8)		//should never be more than 16, but eh. Recursive style
	{
		//log(ALWAYS)<<"Writing big number: more than 8 bits ("<<bitNum<<"): "<<myInt<<endLog();
		unsigned int quotient = myInt/256;
		unsigned int rmdr = myInt - 256 * quotient;
		writeBits(quotient,bitNum-8);	//write the most important bits first (those above the rightmost 8 bits)
		writeSingleByte((unsigned char)rmdr);	//write remaining 8 bits
	}
	else				//if the int fits on one byte, don't bother and call the original function that works on a byte
	{
		//log(ALWAYS)<<"writeBits: calling char version ("<<bitNum<<"): "<<myInt<<endLog();
		unsigned char myByte = (unsigned char)myInt;
		writeBits(myByte,bitNum);
	}
}


void BitArray::padLastByte()	//pad last byte with 1s (and add it).
{
	if(waitingBits>0)
	{
		unsigned char padding = std::pow(2,8-waitingBits)-1;
		incompleteByte=incompleteByte<<(8-waitingBits);
		incompleteByte+=padding;
		writtenBytes.push_back(incompleteByte);
		incompleteByte=0;
		waitingBits=0;
	}
}

void BitArray::writeBytes(const unsigned char myBytes[])
{
	for(unsigned int i=0;i<sizeof(myBytes)/sizeof(myBytes[0]);++i)
	{
		writeSingleByte(myBytes[i]);
	}
}

void BitArray::write2Bytes(unsigned char myBytes[])
{
	for(int i=0;i<2;++i)
	{
		writeSingleByte(myBytes[i]);
	}
}

void BitArray::writeBytes(const std::vector<unsigned char> &myBytes)
{
	for(int i=0;i<myBytes.size();++i)
	{
		writeSingleByte(myBytes[i]);
	}
}

unsigned char BitArray::readOneByte()
{
	return readBitsChar(8);
}

//For bitNum > 8, return value in a int
unsigned int BitArray::readBitsInt(int bitNum)
{
	if(bitNum>8)
	{
		unsigned int myInt;
		myInt = (unsigned int)readBitsChar(8);
		myInt = myInt << (bitNum-8);
		myInt+=readBitsInt(bitNum-8);
		return myInt;
	}
	else
	{
		return (unsigned int)readBitsChar(bitNum);
	}
}

//returns value on a single byte. bitNum must be <=8
unsigned char BitArray::readBitsChar(int bitNum)
{
	unsigned char myChar;
	//log(ALWAYS)<<"Reading bits."<<endLog();
	//log(ALWAYS)<<"bitNum = "<<bitNum<<endLog();
	//log(ALWAYS)<<"Current state: readingByteIndex = "<<readingByteIndex<<", readingBitOffset = "<<readingBitOffset<<endLog();
	//log(ALWAYS)<<"Current byte = "<<writtenBytes[readingByteIndex]<<endLog();
	if(bitNum+readingBitOffset<=8)
	{
		unsigned char curByte = writtenBytes[readingByteIndex];
		//curByte = (unsigned char) (int(curByte) % std::pow(2,8-readingBitOffset));
		//log(ALWAYS)<<"curByte = "<<(unsigned int)(curByte)<<endLog();
		//log(ALWAYS)<<"division: "<<(unsigned int)((curByte/std::pow(2,8-readingBitOffset)))<<endLog();
		//log(ALWAYS)<<"remultiplied: "<<(unsigned int)(curByte/std::pow(2,8-readingBitOffset)) * std::pow(2,8-readingBitOffset)<<endLog();
		curByte = curByte - (unsigned int)(curByte/std::pow(2,8-readingBitOffset)) * std::pow(2,8-readingBitOffset);	//get remainder
		//log(ALWAYS)<<"curByte(2) = "<<(unsigned int)(curByte)<<endLog();
		readingBitOffset+=bitNum;
		myChar = (curByte>>(8-readingBitOffset));
		//log(ALWAYS)<<"myChar = "<<myChar<<endLog();
		//log(ALWAYS)<<"myChar (int) = "<<int(myChar)<<endLog();
		if(readingBitOffset==8)
		{
			readingBitOffset=0;
			++readingByteIndex;
		}
	}
	else	//> 8
	{
		int initialBitOffset = readingBitOffset;
		myChar = readBitsChar(8-initialBitOffset);	//finish the current byte
		myChar = myChar<<(bitNum-8+initialBitOffset);	//leave room for the remaining bits
		myChar+= readBitsChar(bitNum-8+initialBitOffset);
	}

	return myChar;
}

void BitArray::moveBackOneByte()
{
	moveBackNBits(8);
}

void BitArray::moveBackNBits(int bitNum)
{
	readingBitOffset-=bitNum;
	while(readingBitOffset<0)
	{
		readingBitOffset+=8;
		--readingByteIndex;
	}
}

void BitArray::initializeReading()
{
	readingByteIndex=0;
	readingBitOffset=0;
}

unsigned char BitArray::readEndOfByte()
{
	if (readingBitOffset>0)
	{
		return readBitsChar(8-readingBitOffset);
	}
	else
	{
		return 0;
	}
}

void HuffNode::increment(int depth)
{
	//log(ALWAYS)<<"increment depth = "<<depth<<endLog();
	if(this->isLeaf)
	{
		//log(ALWAYS)<<"adding bit"<<endLog();
		*(this->bitsNum)+=1;
	}
	else if((this-left)&&(this->right))
	{
		//log(ALWAYS)<<"incrementing left"<<endLog();
		(this->left)->increment(depth+1);
		//log(ALWAYS)<<"incrementing right"<<endLog();
		(this->right)->increment(depth+1);
	}
	/*if(this->left)
	{
		log(ALWAYS)<<"incrementing left"<<endLog();
		(this->left)->increment();
	}
	if(this->right)
	{
		log(ALWAYS)<<"incrementing right"<<endLog();
		(this->right)->increment();
	}
	if(this->bitsNum)
	{
		log(ALWAYS)<<"adding bit"<<endLog();
		*(this->bitsNum)+=1;
	}*/
}

void HuffTree::addOccurences(unsigned char myCode, int occurencesNum)
{
	int myInd = int(myCode);
	this->occurences[myInd]+=occurencesNum;
}

void HuffTree::computeHuffmanTable()
{
	//create list of nodes
	std::list<HuffNode*> nodeList;

	int totalOccurence=0;
	//create node for each byte with non-null occurences. Add it to the list
	for(int i=0;i<256;++i)
	{
		if(this->occurences[i]>0)
		{
			HuffNode* test = new HuffNode(&(this->bitsNum[i]),this->occurences[i]);
			nodeList.push_back(test);
			totalOccurence+=this->occurences[i];
		}

	}
	
	//loop
	while(nodeList.size()>1)	//when all nodes are grouped (one node left), we're done
	{
		//take two lowest proba
		float proba1 = float(totalOccurence);
		float proba2 = float(totalOccurence);
		HuffNode* node1;
		HuffNode* node2;
		std::list<HuffNode*>::iterator it = nodeList.begin();
		std::list<HuffNode*>::iterator it1;
		std::list<HuffNode*>::iterator it2;

		while(it!=nodeList.end())
		{
			if(proba1>proba2)
			{
				if((*it)->getProba()<proba1)
				{
					proba1 = (*it)->getProba();
					it1=it;
				}
			}
			else if((*it)->getProba()<proba2)
			{
				proba2 = (*it)->getProba();
				it2=it;
			}
			++it;
		}
		//make node with those two
		node1 = *it1;
		node2 = *it2;
		
		HuffNode* test = new HuffNode(node1,node2);
		
		//increment node
		//log(ALWAYS)<<"Incrementing node"<<endLog();
		test->increment(0);
		//log(ALWAYS)<<"pushing new node in list"<<endLog();
		nodeList.push_back(test);
		//log(ALWAYS)<<"erasing nodes"<<endLog();
		
		//remove them - take care of removing order that can mess up the iterators
		if(std::distance(nodeList.begin(),it1)>std::distance(nodeList.begin(),it2))
		{
			//log(ALWAYS)<<"erasing 1st node (it1)"<<endLog();
			it1=nodeList.erase(it1);
			//log(ALWAYS)<<"erasing 2nd node (it2)"<<endLog();
			it2=nodeList.erase(it2);
		}
		else
		{
			//log(ALWAYS)<<"erasing 1st node (it2)"<<endLog();
			it2=nodeList.erase(it2);
			//log(ALWAYS)<<"erasing 2nd node (it1)"<<endLog();
			it1=nodeList.erase(it1);
		}
	}

	computeCodedBytes();
	
}

void HuffTree::computeCodedBytes()
{
	int maxBitNum=0;
	for(int i=0;i<256;++i)
	{
		if(bitsNum[i]>0)
		{
			maxBitNum=std::max(maxBitNum,bitsNum[i]);
			//log(ALWAYS)<<"code word "<<i<<": "<<occurences[i]<<" occurences"<<endLog();
			//log(ALWAYS)<<"code word "<<i<<": "<<bitsNum[i]<<" bits"<<endLog();
		}
	}

	//Compute coded bits from bit lengths
	int curBitNum=1;
	int curCode=0;
	while(curBitNum<=maxBitNum)		//process bit lengths in increasing order
	{
		for(int i=0;i<256;++i)		//we process code words in the 'natural' order given by their numerical value
		{
			if(bitsNum[i]==curBitNum)
			{
				codedBytes[i]=curCode;
				curCode+=1;
			}
		}
		curCode=curCode<<1;		//works on int?
		++curBitNum;
	}

	for(int i=0;i<256;++i)
	{
		if(bitsNum[i]>0)
		{
			int rmdr=codedBytes[i];
			maxBitNum=std::max(maxBitNum,bitsNum[i]);
			//log(ALWAYS)<<"code word "<<i<<": "<<bitsNum[i]<<" bits (";
			//log(ALWAYS)<<"code word "<<i<<": (";
			for(int j=0;j<bitsNum[i];++j)
			{
				int currentBit = int(rmdr/std::pow(2,bitsNum[i]-1-j));
				rmdr-= currentBit*std::pow(2,bitsNum[i]-1-j);
				//log(ALWAYS)<<currentBit;
			}
			//log(ALWAYS)<<")"<<endLog();
		}
	}
}