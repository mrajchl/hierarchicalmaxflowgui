#include <iostream>
#include <QApplication>
#include "MainApplication.h"


int main( int argc, char *argv[] ) 
{
	
QApplication::setColorSpec(QApplication::ManyColor);
QApplication app(argc, argv);
app.setStyle( "plastique" ); 
app.setStyle("windowsxp"); 
MainApplication mainWin;

mainWin.show();

return app.exec();

}
