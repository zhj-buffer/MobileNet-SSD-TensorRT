/******************************************************
 * FileName:      main.c
 * Company:       ????
 * Date:          2016/07/09  09:30
 * Description:   Lobot???????????,??stc89c52rc??
 *                @@????:????????????9600???@@
 *****************************************************/

#include "uart.h"
#include "LobotSerialServo.h"
#include "main.h"

int move_back(int speed)
{
	    LobotSerialMotorMove(2, speed * -1 );
	    LobotSerialMotorMove(3, speed );
	    usleep(1000 * 1000);
	    LobotSerialMotorMove(2, 0 );
	    LobotSerialMotorMove(3, 0 );
}
int move_forward(int speed)
{
	    LobotSerialMotorMove(2, speed );
	    LobotSerialMotorMove(3, speed * -1 );
	    usleep(1000 * 1000);
	    LobotSerialMotorMove(2, 0 );
	    LobotSerialMotorMove(3, 0 );
}
int move_left_shift(int speed)
{
	float turn = 0.45;
//	speed = (float)(speed) * turn;
	printf("speed: %d\n", (int)(speed * turn));
	    LobotSerialMotorMove(2, (speed * -1) * turn);
	    LobotSerialMotorMove(3, (speed * -1) * turn);
	    LobotSerialMotorMove(1, speed );
}
int move_right_shift(int speed)
{
	float turn = 0.45;
	    LobotSerialMotorMove(2, speed * turn);
	    LobotSerialMotorMove(3, speed * turn );
	    LobotSerialMotorMove(1, speed * -1 );
}
int move_left(int speed)
{
	float turn = 0.45;
//	speed = (float)(speed) * turn;
	printf("speed: %d\n", (int)(speed ));
	    LobotSerialMotorMove(1, speed );
	    LobotSerialMotorMove(2, speed );
	    LobotSerialMotorMove(3, speed );
//	    usleep(1000 * 1000);
}
int move_right(int speed)
{
	float turn = 0.45;
	    LobotSerialMotorMove(1, speed * -1 );
	    LobotSerialMotorMove(2, speed * -1 );
	    LobotSerialMotorMove(3, speed * -1 );
}
int move_stop()
{
	    LobotSerialMotorMove(1, 0 );
	    LobotSerialMotorMove(2, 0 );
	    LobotSerialMotorMove(3, 0 );
}
int move_single(int speed, int id)
{
	    LobotSerialMotorMove(id, speed * -1 );
}
void set_id(int id)
{
    LobotSerialServoSetID(3, id);
}

int move_init()
{
	return uart_init();
}
int move_deinit()
{
	uart_deinit();
}

#if 0
int main()
{
        printf("%s, %s, %d\n",__func__,__FILE__,__LINE__);
	int ret = uart_init();
        printf("ret: %d %s, %s, %d\n",ret, __func__,__FILE__,__LINE__);
    if (ret < 0) {
        uart_deinit();
        return -1;
    }

   //set_id(1);
   //move_single(1000, 1);
   //return 0;

    move_forward(1000);
    move_back(1000);
    move_left(1000);
    move_right(1000);
    move_left_shift(1000);
    move_right_shift(1000);
 
//    LobotSerialServoStop(2);
//    LobotSerialServoUnload(2);
//    LobotSerialServoMove(1, 880, 2000);

	uart_deinit();

	return 0;

    for (int i = 0; i < 254; i++) {
        LobotSerialServoMove(i, 1800, 5000);
    }
        LobotSerialServoMove(2, 500, 100);
        LobotSerialServoMove(3, 500, 100);

//while(1)
	{
		//LobotSerialServoSetID(1, 2);
        LobotSerialServoMove(1, 500, 100);
        
        usleep(100 * 1000);
        printf("%s, %s, %d\n",__func__,__FILE__,__LINE__);
	}

    uart_deinit();
}
#endif
