//
// Created by phs on 21. 6. 10..
//
#include <mutex>
#include "TimeClock.h"

namespace SuperPoint
{
    std::unique_ptr<TimeClock> TimeClock::instance = nullptr;
    std::once_flag TimeClock::flag;

    TimeClock &TimeClock::GetInstance(void)
    {
        std::call_once(TimeClock::flag, []()
        {
            TimeClock::instance.reset(new TimeClock);
        });
        return *TimeClock::instance;
    }


    void TimeClock::SetStart(const std::string comment)
    {
        std::cout << "--[" << comment << "]------------------------------" << std::endl;
        start = clock();
    }

    void TimeClock::SetEnd()
    {
        finish = clock();
        print_duration();
    }
}