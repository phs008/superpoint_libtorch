//
// Created by phs on 21. 6. 10..
//
#include <iostream>
#include <ctime>
#include <memory>

namespace SuperPoint
{
    class TimeClock
    {
    public:
        static TimeClock& GetInstance(void);
    private:
        TimeClock() = default;
        static std::once_flag flag;

    protected:
        static std::unique_ptr<TimeClock> instance;
    private:
        std::clock_t start, finish;
    public:
        void SetStart(const std::string comment = "");
        void SetEnd();
        void print_duration()
        {
            double duration = (double) (finish - start) / CLOCKS_PER_SEC;
            std::cout << duration << "s" << std::endl;
            std::cout << "------------------------------------------------" << std::endl;
        }
    };
};
