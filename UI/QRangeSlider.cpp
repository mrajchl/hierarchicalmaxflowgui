#include "QRangeSlider.h"
//Added by qt3to4:
#include <QApplication>
#include <QPaintEvent>
#include <QResizeEvent>
#include <QMouseEvent>
#include <iostream>
#include <QtGui>
#include <QStyle>
#include <QMotifStyle>

using namespace Qt;

template <typename T> T mapRange(const T& val, const T& srcMin, const T& srcMax, 
								 const T& dstMin, const T& dstMax)
{

	return ((val-srcMin)*(dstMax-dstMin)/(srcMax-srcMin)+dstMin);
}
//////////////////////////////////////////////////////////////////////////
QRangeSlider::QRangeSlider(QWidget *parent,Qt::Orientation o,  Qt::WFlags wFlags )
    : mOffset(15), mOrient(o), mMin(0), mMax(100), mLower(0), mUpper(100), QWidget( parent, wFlags )
{  
	font.setPointSize( 9 );
	int w,h;

	if(mOrient==Qt::Vertical){
		w=27;
		h=122;
	}
	else{
		w=30;
		h=30;
	}

	this->setMinimumSize(w,h);
	segSelected=UNKNOWN;

	resize( QSize(w, h).expandedTo(minimumSizeHint()) );
    
	QSizePolicy sp( QSizePolicy::Expanding, QSizePolicy::Fixed );

    if ( mOrient == Qt::Vertical )
		sp.transpose();
    
	setSizePolicy( sp );
   
};
//////////////////////////////////////////////////////////////////////////
QRangeSlider::~QRangeSlider()
{}
//////////////////////////////////////////////////////////////////////////
void QRangeSlider::paintEvent(QPaintEvent *e )
{ 
	QString ll;
	QString uu;
	ll = QString::number( mLower );
	uu= QString::number(mUpper);
		
	QRect cr(0, 0, width()-1, height()-1);

	QPainter p(this);	
	p.setRenderHint(QPainter::Antialiasing);
	p.fillRect(rect(), Qt::blue);

	//QBrush brush = backgroundPixmap() ? QBrush(backgroundColor(), *backgroundPixmap() ) 	: QBrush( backgroundColor() );

//USAF		QBrush brush = backgroundPixmap() ? QBrush(Qt::blue, *backgroundPixmap() ) : QBrush( Qt::blue );
QBrush brush=QBrush( Qt::blue );



	p.fillRect( cr, brush );
 
	if(mOrient==Qt::Horizontal)	
	{	
		QStyleOptionButton option;
		option.state=QStyle::State_Sunken;

		option.rect=rect();
		style()->drawPrimitive(QStyle::PE_PanelButtonCommand, &option, &p, this);

	
		p.setBrush(QBrush(Qt::lightGray, Qt::Dense4Pattern));
		p.drawRect(rm.x(), rm.y()+2, rm.width(), rm.height()-4);

//USAF		qDrawShadeLine(&p, 0, height()/2, width(), height()/2, colorGroup(), TRUE, 2);

		option.state=QStyle::State_Raised|QStyle::State_Enabled;
		option.rect=rl;
		style()->drawPrimitive(QStyle::PE_PanelButtonCommand, &option, &p, this);


		option.state=QStyle::State_Raised|QStyle::State_Enabled;
		option.rect=ru;
		style()->drawPrimitive(QStyle::PE_PanelButtonCommand, &option, &p, this);

//USAF		p.setPen ( colorGroup().text() );
		//p.setFont( font );
//		p.setFont(QFont("Arial", 9));

		p.drawText ( rm, Qt::AlignLeft|Qt::AlignVCenter, ll); 
		p.drawText ( rm, Qt::AlignRight|Qt::AlignVCenter, uu); 
	}
	else
	{
		QStyleOptionButton option;
		option.state=QStyle::State_Sunken;
		option.rect=rect();
		style()->drawPrimitive(QStyle::PE_PanelButtonCommand, &option, &p, this);

		p.setBrush(QBrush(Qt::lightGray, Qt::Dense4Pattern));
		p.drawRect(rm.x()+2, rm.y(), rm.width()-4, rm.height());

//USAF		qDrawShadeLine(&p, width()/2, 0, width()/2, height(), colorGroup(), TRUE, 2);	

		option.state=QStyle::State_Raised|QStyle::State_Enabled;
		option.rect=rl;
		style()->drawPrimitive(QStyle::PE_PanelButtonCommand, &option, &p, this);

		option.state=QStyle::State_Raised|QStyle::State_Enabled;
		option.rect=ru;
		style()->drawPrimitive(QStyle::PE_PanelButtonCommand, &option, &p, this);

//USAF		p.setPen ( colorGroup().text() );
//		p.setFont( font );
		p.drawText ( rm, Qt::AlignTop|Qt::AlignHCenter, ll); 
		p.drawText ( rm, Qt::AlignBottom|Qt::AlignHCenter, uu); 
	}


	p.end();

} 
//////////////////////////////////////////////////////////////////////////
void QRangeSlider::mousePressEvent(QMouseEvent *e)
{
	QPoint p=e->pos();
	lastPos=p;
	switch( e->button() )
	{
		case Qt::LeftButton:
			{
			whichSegment(p);
			
				break;
			}
		case Qt::MidButton:
			{

			break;
			}
		case Qt::RightButton:
			{
			break;
			}
			default:
			break;
	}
}
//////////////////////////////////////////////////////////////////////////
void QRangeSlider::mouseMoveEvent(QMouseEvent *e)
{
QPoint p=e->pos();	
switch( e->buttons())
{
	case Qt::LeftButton:
		{
	
		if(segSelected==Lower)moveLower(p);
		else if(segSelected==Middle)moveMiddle(p);
		else if(segSelected==Upper)moveUpper(p);
	
			break;
		}
	case Qt::MidButton:
		{

		break;
		}
	case Qt::RightButton:
		{
		break;
		}
	default:
		break;
	}
lastPos=p;
}
//////////////////////////////////////////////////////////////////////////
void QRangeSlider::mouseReleaseEvent(QMouseEvent *e)
{
	QPoint p=e->pos();		
	switch( e->buttons() )
	{
		case Qt::LeftButton:
		{
			break;
		}
		case Qt::MidButton:
		{
			break;
		}
		case Qt::RightButton:
		{
			break;
		}
		
		default:
			break;
	}
	//emit mouseRelease();
	segSelected=UNKNOWN;
}
//////////////////////////////////////////////////////////////////////////
QSize QRangeSlider::minimumSizeHint() const
{
	int w,h;

	if( mOrient == Qt::Horizontal ){
		w=120;
		h=20;
	}else{
		w=20;
		h=120;
	}

	QSize s = sizeHint();
	s.setHeight( h );
	s.setWidth( w );
	
	return s;
}
//////////////////////////////////////////////////////////////////////////
void QRangeSlider::setMinValue(int a)
{
	this->mMin=a;
};
//////////////////////////////////////////////////////////////////////////
void QRangeSlider::setMaxValue(int a)
{
	this->mMax=a;
};	
//////////////////////////////////////////////////////////////////////////
void QRangeSlider::setLower(int a)
{
	this->mLower=a;
};	
//////////////////////////////////////////////////////////////////////////
void QRangeSlider::setUpper(int a)
{
	this->mUpper=a;
};	
//////////////////////////////////////////////////////////////////////////
void QRangeSlider::whichSegment(QPoint& p)
{
	segSelected=UNKNOWN;
	if(rl.contains(p,TRUE))segSelected=Lower;
	if(rm.contains(p,TRUE))segSelected=Middle;
	if(ru.contains(p,TRUE))segSelected=Upper;
}
//////////////////////////////////////////////////////////////////////////
void QRangeSlider::moveLower(QPoint& p)
{
	QRect l=rl;
	QRect m=rm;
	QRect u=ru;

	QRect wr=this->rect();
	QPoint c=l.center();
	QPoint dd=p-c;

	if(mOrient==Qt::Horizontal)
	{
		int oy=l.y();
		int oh=l.height();

		l.moveCenter(p);
		l.setY(oy);
		l.setHeight(oh);

		if( !wr.contains(l,TRUE) || l.intersects(u) || l.topRight().x() >= u.topLeft().x())
			return ;

		m.setTopLeft(l.topRight());
		m.setBottomLeft(l.bottomRight());
		m.setTopRight(u.topLeft());
		m.setBottomRight(u.bottomLeft());

		m.setY(oy);
		u.setY(oy);

		m.setHeight(oh);
		u.setHeight(oh);

		l=l.normalized();
		m=m.normalized();
		u=u.normalized();

		rl=l;
		rm=m;
		ru=u;
	}
	else{
		int ox=l.x();
		int ow=l.width();

		l.moveCenter(p);
		l.setX(ox);
		l.setWidth(ow);

		if( !wr.contains(l,TRUE) || l.intersects(u) || l.bottomLeft().y() >= u.topLeft().y())
			return;

		m.setTopLeft(l.bottomLeft());
		m.setBottomLeft(u.topLeft());
		m.setTopRight(l.bottomRight());
		m.setBottomRight(u.topRight());

		m.setX(ox);
		u.setX(ox);

		m.setWidth(ow);
		u.setWidth(ow);

		l=l.normalized();
		m=m.normalized();
		u=u.normalized();

		rl=l;
		rm=m;
		ru=u;
	}
	repaint();
	updateRange();
}
//////////////////////////////////////////////////////////////////////////
void QRangeSlider::moveMiddle(QPoint& p)
{
	QRect l=rl;
	QRect m=rm;
	QRect u=ru;

	QRect wr=this->rect();
	QPoint c=m.center();
//	QPoint dd=p-c;
	QPoint dd=p-lastPos;

	if(mOrient==Qt::Horizontal)
	{
		int oy=m.y();
		int oh=m.height();

//		m.moveCenter(p);
//		l.moveCenter(l.center()+dd);
//		u.moveCenter(u.center()+dd);

		m.translate(dd.x(),0);
		l.translate(dd.x(),0);
		u.translate(dd.x(),0);

		l.setY(oy);
		m.setY(oy);
		u.setY(oy);

		l.setHeight(oh);
		m.setHeight(oh);
		u.setHeight(oh);

		if( !wr.contains(l,TRUE) || !wr.contains(m,TRUE) || !wr.contains(u,TRUE))
			return ;

		l=l.normalized();
		m=m.normalized();
		u=u.normalized();

		rl=l;
		rm=m;
		ru=u;
	}
	else{
		int ox=l.x();
		int ow=l.width();

//		m.moveCenter(p);
//		l.moveCenter(l.center()+dd);
//		u.moveCenter(u.center()+dd);

		m.translate(0, dd.y());
		l.translate(0, dd.y());
		u.translate(0, dd.y());


		l.setX(ox);
		m.setX(ox);
		u.setX(ox);

		l.setWidth(ow);
		m.setWidth(ow);
		u.setWidth(ow);

		if( !wr.contains(l,TRUE) || !wr.contains(m,TRUE) || !wr.contains(u,TRUE))
			return ;

		l=l.normalized();
		m=m.normalized();
		u=u.normalized();

		rl=l;
		rm=m;
		ru=u;
	}
	repaint();
	updateRange();
}
//////////////////////////////////////////////////////////////////////////
void QRangeSlider::moveUpper(QPoint& p)
{
	QRect l=rl;
	QRect m=rm;
	QRect u=ru;

	QRect wr=this->rect();
	QPoint c=l.center();
	QPoint dd=p-c;

	if(mOrient==Qt::Horizontal)
	{
		int oy=u.y();
		int oh=u.height();
		u.moveCenter(p);
		u.setY(oy);
		u.setHeight(oh);
		if( !wr.contains(u,TRUE) || u.intersects(l) || u.topLeft().x() <= l.topRight().x() )
			return ;

		m.setTopLeft(l.topRight());
		m.setBottomLeft(l.bottomRight());
		m.setTopRight(u.topLeft());
		m.setBottomRight(u.bottomLeft());

		l.setY(oy);
		m.setY(oy);

		l.setHeight(oh);
		m.setHeight(oh);

		l=l.normalized();
		m=m.normalized();
		u=u.normalized();

		rl=l;
		rm=m;
		ru=u;
	}
	else{
		int ox=l.x();
		int ow=l.width();

		u.moveCenter(p);
		u.setX(ox);
		u.setWidth(ow);
		if( !wr.contains(u,TRUE) || u.intersects(l) || u.topLeft().y() <= l.bottomLeft().y() )
			return ;

		m.setBottomLeft(u.topLeft());
		m.setBottomRight(u.topRight());

		l.setX(ox);
		m.setX(ox);

		l.setWidth(ow);
		m.setWidth(ow);

		l=l.normalized();
		m=m.normalized();
		u=u.normalized();

		rl=l;
		rm=m;
		ru=u;
	}
	repaint();
	updateRange();
}
//////////////////////////////////////////////////////////////////////////
void QRangeSlider::updateRange()
{
	int ww,l,u;
	if( mOrient == Qt::Horizontal )
	{
		ww=width()-2;	
			
		l=mapRange(rl.topRight().x(),mOffset,ww-mOffset,mMin,mMax);	
		u=mapRange(ru.topLeft().x(),mOffset,ww-mOffset,mMin,mMax);	
			
		}
	else
	{
		ww=height()-2;	
			
		l=mapRange(rl.bottomLeft().y() ,mOffset,ww-mOffset,mMin,mMax);	
		u=mapRange(ru.topLeft().y(),mOffset,ww-mOffset,mMin,mMax);	
	}
	
	if(l!=mLower || u!=mUpper)
	{
		if(l<mMin)l=mMin;
		if(u>mMax)u=mMax;

		mLower=l;
		mUpper=u;
		//std::cout<<"l="<<l<<" u="<<u<<std::endl;
		emit valueChanged(mLower,mUpper);
	}
}
//////////////////////////////////////////////////////////////////////////
void QRangeSlider::updateRect()
{
	int ww=width()-2;
	int hh=height()-2;

	int l,u;

	if( mOrient == Qt::Horizontal )
	{
		l=mapRange(mLower, mMin,mMax, mOffset,ww-mOffset);	
		u=mapRange(mUpper, mMin,mMax, mOffset,ww-mOffset);	

		rl.setRect (l-mOffset,1, mOffset, hh); 
		rm.setRect (l+1,1,u-l, hh ); 
		ru.setRect(u,1,mOffset,hh);
	}
	else
	{
		l=mapRange(mLower, mMin,mMax, mOffset,hh-mOffset);	
		u=mapRange(mUpper, mMin,mMax, mOffset,hh-mOffset);	


		rl.setRect(1,l-mOffset,ww,mOffset);
		rm.setRect (1,l+1,ww, u-l ); 
		ru.setRect (1,u, ww,mOffset); 
	}
}

//////////////////////////////////////////////////////////////////////////
void QRangeSlider::resizeEvent( QResizeEvent * e)
{
	this->updateRect();
}

