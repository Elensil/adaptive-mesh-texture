/* *************************************************
 * Copyright (C) 2014 INRIA
 * *************************************************/
#ifndef DUAL_STREAM_H
#define DUAL_STREAM_H

// adapted from:
// http://stackoverflow.com/questions/1760726/how-can-i-compose-output-streams-so-output-goes-multiple-places-at-once

// Class to split between 2 std::ostream's
class DualStream: public std::ostream
{
protected:
	
	struct DualBuffer : public std::streambuf
	{
		void setBuffer(const unsigned int pos, std::streambuf* buf)
		{
			bufs[pos] = buf ;
		}

		virtual int overflow(int c)
		{
			bufs[0]->sputc(c) ;
			bufs[1]->sputc(c) ;
			return c ;
		}

		private:
			std::streambuf* bufs[2] ;

	} ;
	DualBuffer myBuffer ;
	
public:
	
	DualStream() : std::ostream(NULL)
	{
		std::ostream::rdbuf(&myBuffer) ;
	}
	
	// Link the given stream
	void link(const unsigned int pos, std::ostream& out)
	{
		out.flush() ;
		myBuffer.setBuffer(pos, out.rdbuf()) ;
	}
} ;

#endif // #ifndef DUAL_STREAM_H

