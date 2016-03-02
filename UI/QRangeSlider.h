/********************************************************************
	created:	2005/05/08
	file base:	QRangeSlider
	file ext:	h
	author:		Usaf E. Aladl
	
	purpose:	Window and Level slider
*********************************************************************/
#ifndef _Q_RANGE_SLIDER_H_
#define _Q_RANGE_SLIDER_H_
 #include <QRect>
#include <QWidget>
#include <QResizeEvent>
#include <QMouseEvent>
#include <QPaintEvent>
class QRect;

class QRangeSlider : public QWidget
{
	Q_OBJECT	

public:
	/**  constructor */
	QRangeSlider( QWidget *parent=0,Qt::Orientation o=Qt::Horizontal, Qt::WFlags wFlags=0 );
	/**  destructor */
	~QRangeSlider();
	/** set the widget orientation (Vertical or Horizontal) **/
	void setOrientation(Qt::Orientation o){this->mOrient=o;};
	QSize minimumSizeHint() const;
	int lower() const{return this->mLower;};
	int upper() const{return this->mUpper;};
	int minRangeSlider() const{return this->mMin;};
	int maxRangeSlider() const{return this->mMax;};	

public slots:
	 void setMinValue(int);
	 void setMaxValue(int);	
	 void setLower(int);
	 void setUpper(int);

protected:
	enum Segment{ Lower, Middle, Upper, UNKNOWN };

	virtual void paintEvent( QPaintEvent * e);
	virtual void mousePressEvent(QMouseEvent *e);
	virtual void mouseMoveEvent(QMouseEvent *e);
	virtual void mouseReleaseEvent(QMouseEvent *e);	
	void resizeEvent(QResizeEvent*);
	void whichSegment(QPoint& p);
	void moveLower(QPoint& p);
	void moveMiddle(QPoint& p);
	void moveUpper(QPoint& p);
	void updateRange();
	void updateRect();


private:
	Qt::Orientation	mOrient;
	QRect rl;
	QRect rm;
	QRect ru;
	int mOffset;
	int mMin;
	int mMax;
	int mLower;
	int mUpper;
	Segment segSelected;
//USAF QColorGroup cg;
	QPoint lastPos;
	QFont font;

signals:
    void valueChanged(int ,int);	
};

#endif