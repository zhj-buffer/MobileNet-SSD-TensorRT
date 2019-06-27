/******************************************************
 * FileName:      uart.h
 * Company:       ????
 * Date:           2016/07/08  20:02
 * Description:    Lobot???????????,??stc89c52rc??,
 *                 ???????????????????????
 *****************************************************/

#ifndef UART_H_
#define UART_H_
#include<stdio.h>
#include<stdlib.h>
#include<fcntl.h>
#include<unistd.h>
#include<assert.h>
#include<termios.h>
#include<string.h>
#include<sys/time.h>
#include<sys/types.h>
#include<errno.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>


bool isRxCompleted(void);

extern int uart_init(void);
extern int uart_deinit(void);
extern int uart_close(int fd);
extern int uart_write(int fd,const char *w_buf,size_t len);
extern int uart_read(int fd,char *r_buf,size_t len);
//extern int uartWriteBuf(unsigned char *buf, size_t len);
//extern int uartReadBuf(unsigned char *buf, size_t len);
extern int LobotSerialWrite(unsigned char *buf, size_t len);
extern int LobotSerialRead(unsigned char *buf, size_t len);


#endif
